"""RLExecution: wraps a trained PPO actor as an ExecutionAlgo.

At each decision the same 8-dim observation the env was trained on is built
from a LiveFeatures tracker; the argmax action a maps to a market child of
``(a/5) * twap_slice`` — identical semantics to ExecutionEnv.step, so the
policy transfers without modification.
"""

from __future__ import annotations

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.execution.live import LiveFeatures
from orderflow.execution.types import ChildOrder, ExecState, ExecutionTask
from orderflow.registry import register_execution


@register_execution("rl")
class RLExecution:
    name = "rl"

    def __init__(self, policy=None) -> None:
        # policy: ActorCritic (or anything with .act_deterministic); None => TWAP-like
        self.policy = policy
        self.live = LiveFeatures()
        self._prev_mid = 0.0

    def load_policy(self, path: str, obs_dim: int = 8, n_actions: int = 11):
        import torch

        from orderflow.execution.rl.ppo import ActorCritic

        net = ActorCritic(obs_dim, n_actions)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        self.policy = net
        return self

    def reset(
        self, task: ExecutionTask, book: LimitOrderBook, rng: np.random.Generator
    ) -> None:
        self.task = task
        self.live = LiveFeatures(tick_size=book.tick_size)
        self.live.update(book)
        mid = book.mid()
        self._prev_mid = float(mid) if mid is not None else 0.0
        self.n_steps = max(1, round(task.horizon / task.step))
        self.twap_slice = max(1, round(task.total_qty / self.n_steps))

    def decide(
        self, t: float, book: LimitOrderBook, state: ExecState
    ) -> list[ChildOrder]:
        self.live.update(book, t)
        mid = book.mid()
        recent_ret = float(mid - self._prev_mid) if mid is not None else 0.0
        if mid is not None:
            self._prev_mid = float(mid)
        time_frac = min(state.elapsed / self.task.horizon, 1.0)
        obs = self.live.exec_obs(
            state.remaining_qty / self.task.total_qty, time_frac, recent_ret
        )
        if self.policy is None:
            a = 5  # un-trained policy defaults to the TWAP slice
        else:
            a = int(self.policy.act_deterministic(obs))
        qty = min(round(a / 5.0 * self.twap_slice), state.remaining_qty)
        if qty <= 0:
            return []
        return [ChildOrder(side=self.task.side, qty=qty, kind="market")]
