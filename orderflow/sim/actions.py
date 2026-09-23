"""Actions produced by flow models and consumed by the Simulator."""

from __future__ import annotations

from dataclasses import dataclass

from orderflow.book.types import Side


@dataclass(slots=True)
class SubmitLimit:
    """Limit order priced ``price_offset_ticks`` from the same-side best.

    offset 0 = join the best, negative = improve (inside the spread),
    positive = deeper in the book.
    """

    side: Side
    price_offset_ticks: int
    qty: int
    hidden: bool = False
    display_qty: int | None = None


@dataclass(slots=True)
class SubmitMarket:
    side: Side
    qty: int


@dataclass(slots=True)
class CancelRandom:
    """Cancel a uniformly random resting visible order at ``level_offset``.

    ``level_offset`` 0 = best level on ``side``, 1 = second best, ...
    No-op when the level has no visible orders.
    """

    side: Side
    level_offset: int = 0


@dataclass(slots=True)
class CancelOrder:
    order_id: int


FlowAction = SubmitLimit | SubmitMarket | CancelRandom | CancelOrder
