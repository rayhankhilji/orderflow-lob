"""VWAP: cumulative targets proportional to a historical volume profile.

``VolumeProfile.fit`` averages per-bucket traded volume across calibration runs
and returns the cumulative fraction curve. Because our flow models are
stationary (no intraday U-shape), the fitted profile is near-flat and VWAP
should land close to TWAP — that is the honest result, not a bug.
"""

from __future__ import annotations

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.execution.types import ChildOrder, ExecState, ExecutionTask
from orderflow.registry import register_execution
from orderflow.sim.simulator import SimResult


class VolumeProfile:
    """Cumulative fraction of daily volume done by each decision step."""

    def __init__(self, cum: np.ndarray) -> None:
        # cum[j] = fraction of volume expected executed by step j+1, in (0, 1]
        self.cum = np.asarray(cum, dtype=float)
        self.cum = np.maximum.accumulate(np.clip(self.cum, 0.0, 1.0))
        if len(self.cum):
            self.cum[-1] = 1.0

    @classmethod
    def flat(cls, n_inner: int) -> VolumeProfile:
        return cls(np.linspace(0, 1, max(n_inner, 1) + 1)[1:])

    @classmethod
    def fit(
        cls,
        results: list[SimResult],
        horizon: float,
        step: float,
    ) -> VolumeProfile:
        n_inner = 0
        while (n_inner + 1) * step < horizon - 1e-9:
            n_inner += 1
        n_inner = max(n_inner, 1)
        edges = (np.arange(n_inner) + 1) * step
        bucket_vol = np.zeros(n_inner)
        total = 0.0
        for res in results:
            for f in res.fills:
                j = int(np.searchsorted(edges, f.timestamp, side="right"))
                if j < n_inner:
                    bucket_vol[j] += f.qty
                    total += f.qty
        if total <= 0:
            return cls.flat(n_inner)
        frac = bucket_vol / total
        cum = np.cumsum(frac)
        return cls(cum)

    def target_frac(self, j: int) -> float:
        """Cumulative fraction expected done after decision j (1-indexed)."""
        if len(self.cum) == 0:
            return 1.0
        return float(self.cum[min(max(j - 1, 0), len(self.cum) - 1)])


@register_execution("vwap")
class VWAP:
    name = "vwap"

    def __init__(self, profile: VolumeProfile | None = None) -> None:
        # None => flat profile (equivalent to TWAP); pass a fitted one to differ
        self.profile = profile

    def reset(
        self, task: ExecutionTask, book: LimitOrderBook, rng: np.random.Generator
    ) -> None:
        self.task = task
        self.n_inner = 0
        while (self.n_inner + 1) * task.step < task.horizon - 1e-9:
            self.n_inner += 1
        if self.profile is None:
            self.profile = VolumeProfile.flat(self.n_inner)

    def decide(
        self, t: float, book: LimitOrderBook, state: ExecState
    ) -> list[ChildOrder]:
        j = min(round(state.elapsed / self.task.step), self.n_inner)
        target = round(self.task.total_qty * self.profile.target_frac(j))
        qty = min(target - state.filled_qty, state.remaining_qty)
        if qty <= 0:
            return []
        return [ChildOrder(side=self.task.side, qty=qty, kind="market")]
