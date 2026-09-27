"""Shared types for the execution stack.

Prices are in integer ticks internally; ``cost.py`` converts to dollars with
``book.tick_size``. An :class:`ExecutionTask` describes one parent order: e.g.
sell 100k shares of a $100 stock over 1800 s (~$10M notional).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orderflow.book.types import Fill, Side


@dataclass(slots=True)
class ExecutionTask:
    """One parent order to work through the book.

    ``step`` is the decision interval (seconds): the algo is consulted once per
    step and the remainder is force-liquidated at ``start_time + horizon``.
    """

    side: Side
    total_qty: int
    horizon: float
    step: float
    start_time: float = 0.0


@dataclass(slots=True)
class ChildOrder:
    """A slice the algo wants working right now.

    ``kind="market"`` sweeps the book; ``kind="limit"`` posts at the same-side
    best plus ``price_offset_ticks`` (same convention as ``SubmitLimit``:
    positive = deeper/passive, negative = improve/cross).
    """

    side: Side
    qty: int
    kind: str = "market"
    price_offset_ticks: int = 0


@dataclass(slots=True)
class ExecState:
    """What the algo is told at each decision time."""

    remaining_qty: int
    elapsed: float  # seconds since start_time
    fills: list[Fill] = field(default_factory=list)  # this algo's own fills
    arrival_mid: float = 0.0  # mid (ticks) at start_time

    @property
    def filled_qty(self) -> int:
        return sum(f.qty for f in self.fills)
