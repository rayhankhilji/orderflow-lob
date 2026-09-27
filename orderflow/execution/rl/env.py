"""Gymnasium environment for execution: one decision per ``step`` seconds.

Each env step: submit a market child of size ``(a/5) * twap_slice`` (clipped to
remaining), advance the simulator to the next decision boundary, observe. At
the horizon the remainder is force-liquidated. Reward:

    r = -(dIS_$ / notional_$) * 1e4  -  risk_penalty * remaining_frac^2 * step/horizon

i.e. basis points of notional spent this interval, minus a quadratic penalty on
unfinished inventory (timing risk). The env owns its exec order ids via
``sim.next_id()`` and scans ``book.fills`` itself; ``owner="exec"`` keeps them
attributable in the tape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Order, OrderType, Side
from orderflow.execution.live import LiveFeatures
from orderflow.sim.simulator import Simulator, seed_book

OBS_DIM = 8
N_ACTIONS = 11


class ExecutionEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        flow_factory: Callable,
        seed_kwargs: dict,
        side: Side = Side.SELL,
        total_qty: int = 100_000,
        horizon: float = 1800.0,
        step: float = 10.0,
        tick_size: float = 0.01,
        risk_penalty: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.flow_factory = flow_factory
        self.seed_kwargs = dict(seed_kwargs)
        self.side = side
        self.total_qty = int(total_qty)
        self.horizon = float(horizon)
        self.step_size = float(step)
        self.tick_size = tick_size
        self.risk_penalty = risk_penalty
        self.base_seed = seed
        self.n_steps = max(1, round(horizon / step))
        self.twap_slice = max(1, round(total_qty / self.n_steps))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._episode = 0

    # ------------------------------------------------------------------ api
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = self.base_seed + self._episode if seed is None else seed
        self._episode += 1
        rng = np.random.default_rng(s)
        self.book = LimitOrderBook(tick_size=self.tick_size)
        seed_book(self.book, rng=rng, **self.seed_kwargs)
        self.sim = Simulator(self.book, self.flow_factory(), seed=s)
        self.live = LiveFeatures(tick_size=self.tick_size)
        self.live.update(self.book)
        mid = self.book.mid()
        self.arrival_mid = float(mid) if mid is not None else 0.0
        self._prev_mid = self.arrival_mid
        self.remaining = self.total_qty
        self._step_idx = 0
        self._my_ids: set[int] = set()
        self._fill_cursor = 0
        self._is_dollars = 0.0
        self._is_prev = 0.0
        return self._obs(), {}

    def step(self, action: int):
        a = int(action)
        qty = min(self.remaining, round(a / 5.0 * self.twap_slice))
        if qty > 0:
            self._send_market(qty)
        # advance the world to the next decision boundary
        t_next = min((self._step_idx + 1) * self.step_size, self.horizon)
        while self.sim.t < t_next - 1e-12:
            self.sim.step()
        self._collect_fills()
        self._step_idx += 1
        mid = self.book.mid() or self._prev_mid
        recent_ret = float(mid - self._prev_mid)
        self._prev_mid = float(mid)
        self.live.update(self.book)

        done = self._step_idx >= self.n_steps or self.remaining <= 0
        if done and self.remaining > 0:
            self._send_market(self.remaining)
            self._collect_fills()

        time_frac = min(self.sim.t / self.horizon, 1.0)
        notional = self.arrival_mid * self.tick_size * self.total_qty
        d_is = self._is_dollars - self._is_prev
        self._is_prev = self._is_dollars
        reward = (
            -(d_is / notional) * 1e4
            - self.risk_penalty
            * (self.remaining / self.total_qty) ** 2
            * (self.step_size / self.horizon)
            if notional > 0
            else 0.0
        )
        obs = self._obs(recent_ret, time_frac)
        info = {
            "is_dollars": self._is_dollars,
            "remaining": self.remaining,
            "step_idx": self._step_idx,
        }
        return obs, float(reward), done, False, info

    # --------------------------------------------------------------- intern
    def _obs(self, recent_ret: float = 0.0, time_frac: float = 0.0) -> np.ndarray:
        return self.live.exec_obs(
            self.remaining / self.total_qty, time_frac, recent_ret
        )

    def _send_market(self, qty: int) -> None:
        oid = self.sim.next_id()
        self._my_ids.add(oid)
        self.book.submit(
            Order(
                order_id=oid,
                side=self.side,
                price=None,
                qty=qty,
                remaining=qty,
                timestamp=self.sim.t,
                order_type=OrderType.MARKET,
                owner="exec",
            )
        )

    def _collect_fills(self) -> None:
        s = float(self.side)
        for f in self.book.fills[self._fill_cursor :]:
            if f.taker_id in self._my_ids or f.maker_id in self._my_ids:
                self._is_dollars += (
                    s
                    * (f.price - self.arrival_mid)
                    * f.qty
                    * self.tick_size
                )
                self.remaining -= f.qty
        self._fill_cursor = len(self.book.fills)


def make_exec_env(
    regime: str = "exec", flow: str = "hawkes", seed: int = 0, **env_kwargs
) -> ExecutionEnv:
    """Build an ExecutionEnv on a named regime."""
    from orderflow.sim.regimes import get_regime

    flow_cls, params, seed_kwargs = get_regime(regime, flow=flow)
    return ExecutionEnv(
        flow_factory=lambda: flow_cls(params),
        seed_kwargs=seed_kwargs,
        seed=seed,
        **env_kwargs,
    )
