"""TWAP: equal cumulative targets at each decision time.

The schedule is expressed as cumulative targets c_j = Q * j / M over the M
in-horizon decision steps; each child is ``c_j - filled_so_far`` clipped to
remaining, so missed fills (passive mode) automatically roll forward and the
deadline liquidation is only a safety net.
"""

from __future__ import annotations

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.execution.types import ChildOrder, ExecState, ExecutionTask
from orderflow.registry import register_execution


@register_execution("twap")
class TWAP:
    name = "twap"

    def __init__(self, aggressive: bool = True) -> None:
        self.aggressive = aggressive

    def reset(
        self, task: ExecutionTask, book: LimitOrderBook, rng: np.random.Generator
    ) -> None:
        self.task = task
        self.n_inner = 0
        while (self.n_inner + 1) * task.step < task.horizon - 1e-9:
            self.n_inner += 1

    def _target(self, elapsed: float) -> int:
        j = min(round(elapsed / self.task.step), self.n_inner)
        if self.n_inner <= 0:
            return self.task.total_qty
        return round(self.task.total_qty * j / self.n_inner)

    def decide(
        self, t: float, book: LimitOrderBook, state: ExecState
    ) -> list[ChildOrder]:
        qty = min(self._target(state.elapsed) - state.filled_qty, state.remaining_qty)
        if qty <= 0:
            return []
        return [
            ChildOrder(
                side=self.task.side,
                qty=qty,
                kind="market" if self.aggressive else "limit",
                price_offset_ticks=0,
            )
        ]


@register_execution("twap_passive")
class TWAPPassive(TWAP):
    name = "twap_passive"

    def __init__(self) -> None:
        super().__init__(aggressive=False)
