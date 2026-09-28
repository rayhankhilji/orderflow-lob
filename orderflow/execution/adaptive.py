"""Competing execution variants from the execution-research branch.

``ac_adaptive`` re-solves the Almgren–Chriss trajectory at every decision
step: kappa is recomputed with a live EWMA volatility estimate of the mid,
so the schedule front-loads when vol runs hot and drifts toward TWAP when
calm. Rather than replaying a static plan, each step solves the one-period
problem on *remaining* inventory over *remaining* time:

    n_j = R_j * (1 - sinh(kappa_t (T_rem - tau)) / sinh(kappa_t T_rem))

which is exactly the marginal trade of the re-derived optimal path — the
scheme is a re-planning controller, not a heuristic tilt.

``twap_capped`` clips each TWAP child to a fraction of the *visible*
opposing-side depth (a displayed-liquidity participation cap); unfilled
budget rolls forward through the usual cumulative-target machinery.
"""

from __future__ import annotations

import math

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.execution.almgren_chriss import (
    DEFAULT_ETA,
    DEFAULT_GAMMA,
    DEFAULT_SIGMA,
    ac_kappa,
)
from orderflow.execution.twap import TWAP
from orderflow.execution.types import ChildOrder, ExecState, ExecutionTask
from orderflow.registry import register_execution


@register_execution("ac_adaptive")
class AdaptiveAC:
    """AC re-planned each step with a live EWMA vol estimate."""

    name = "ac_adaptive"

    def __init__(
        self,
        sigma0: float = DEFAULT_SIGMA,
        eta: float = DEFAULT_ETA,
        gamma: float = DEFAULT_GAMMA,
        lam: float = 1e-5,
        ewma_halflife: float = 60.0,
        tick_size: float = 0.01,
        vol_floor_ratio: float = 0.2,
        vol_cap_ratio: float = 5.0,
    ) -> None:
        self.sigma0, self.eta, self.gamma, self.lam = sigma0, eta, gamma, lam
        self.tick_size = tick_size
        # EWMA of squared tick returns, halflife in seconds
        self.decay = 0.5 ** (1.0 / (ewma_halflife))
        self.var_ema = (sigma0 / tick_size) ** 2
        self.vol_floor_ratio, self.vol_cap_ratio = vol_floor_ratio, vol_cap_ratio

    def reset(
        self, task: ExecutionTask, book: LimitOrderBook, rng: np.random.Generator
    ) -> None:
        self.task = task
        self._prev_mid: float | None = book.mid()
        self._prev_t = task.start_time
        self.var_ema = (self.sigma0 / self.tick_size) ** 2

    def _sigma_now(self, t: float, book: LimitOrderBook) -> float:
        mid = book.mid()
        if mid is not None and self._prev_mid is not None and t > self._prev_t:
            ret = mid - self._prev_mid  # ticks
            dt = t - self._prev_t
            # per-second variance contribution of this observation
            inst_var = ret * ret / dt
            w = 1.0 - self.decay**dt
            self.var_ema = (1 - w) * self.var_ema + w * inst_var
        if mid is not None:
            self._prev_mid = mid
        self._prev_t = t
        sigma_ticks = math.sqrt(max(self.var_ema, 1e-12))
        lo = self.vol_floor_ratio * self.sigma0 / self.tick_size
        hi = self.vol_cap_ratio * self.sigma0 / self.tick_size
        return min(max(sigma_ticks, lo), hi) * self.tick_size

    def decide(
        self, t: float, book: LimitOrderBook, state: ExecState
    ) -> list[ChildOrder]:
        t_rem = self.task.horizon - state.elapsed
        if state.remaining_qty <= 0 or t_rem <= self.task.step + 1e-9:
            return []
        sigma = self._sigma_now(t, book)
        kappa = ac_kappa(sigma, self.eta, self.gamma, self.lam, self.task.step)
        if kappa <= 1e-12:
            frac = self.task.step / t_rem  # falls back to TWAP-on-remaining
        else:
            frac = 1.0 - math.sinh(kappa * (t_rem - self.task.step)) / math.sinh(
                kappa * t_rem
            )
        # never slower than TWAP-on-remaining: guarantees the schedule completes
        # without relying on a depth-hungry deadline sweep
        frac = max(frac, self.task.step / t_rem)
        qty = min(math.ceil(state.remaining_qty * frac), state.remaining_qty)
        if qty <= 0:
            return []
        return [ChildOrder(side=self.task.side, qty=qty, kind="market")]


@register_execution("twap_capped")
class CappedTWAP(TWAP):
    """TWAP clipped to a fraction of displayed opposing depth."""

    name = "twap_capped"

    def __init__(self, depth_frac: float = 0.35, depth_levels: int = 8) -> None:
        super().__init__(aggressive=True)
        self.depth_frac, self.depth_levels = depth_frac, depth_levels

    def decide(
        self, t: float, book: LimitOrderBook, state: ExecState
    ) -> list[ChildOrder]:
        orders = super().decide(t, book, state)
        if not orders:
            return orders
        opposing = Side.BUY if self.task.side is Side.SELL else Side.SELL
        depth = sum(q for _, q in book.depth(opposing, self.depth_levels))
        cap = max(1, int(self.depth_frac * depth))
        # the cap may not force the algo behind TWAP-on-remaining, otherwise a
        # depleted-depth stretch leaves an unrecoverable backlog at the deadline
        t_rem = self.task.horizon - state.elapsed
        if t_rem > 0:
            cap = max(cap, math.ceil(state.remaining_qty * self.task.step / t_rem))
        o = orders[0]
        return [
            ChildOrder(
                side=o.side,
                qty=min(o.qty, cap),
                kind=o.kind,
                price_offset_ticks=o.price_offset_ticks,
            )
        ]
