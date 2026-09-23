"""Core types for the order book (prices in integer ticks, qty in integer shares)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np


class Side(IntEnum):
    BUY = 1
    SELL = -1


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(Enum):
    GTC = "gtc"  # rests until filled or cancelled
    IOC = "ioc"  # unfilled remainder cancelled, never rests


@dataclass(slots=True)
class Order:
    order_id: int
    side: Side
    price: int | None  # None for MARKET
    qty: int  # original quantity
    remaining: int  # unfilled quantity
    timestamp: float
    order_type: OrderType = OrderType.LIMIT
    tif: TimeInForce = TimeInForce.GTC
    hidden: bool = False  # fully hidden (not shown in L2 depth, cannot set L1)
    display_qty: int | None = None  # iceberg: visible peak; None => fully displayed
    owner: str = "flow"
    displayed: int | None = None  # iceberg only: currently displayed peak (book-managed)

    @property
    def is_iceberg(self) -> bool:
        return self.display_qty is not None and not self.hidden

    @property
    def visible_remaining(self) -> int:
        """Qty currently shown in L2: peak for iceberg, all for normal, 0 for hidden."""
        if self.hidden:
            return 0
        if self.is_iceberg:
            return self.displayed if self.displayed is not None else 0
        return self.remaining

    @property
    def hidden_remaining(self) -> int:
        """Qty not shown in L2: reserve for iceberg, all for hidden, 0 for normal."""
        if self.hidden:
            return self.remaining
        if self.is_iceberg:
            return self.remaining - self.visible_remaining
        return 0


@dataclass(slots=True)
class Fill:
    timestamp: float
    price: int  # maker's price
    qty: int
    maker_id: int
    taker_id: int
    taker_side: Side
    maker_hidden: bool


class EventType(Enum):
    SUBMIT = "submit"
    CANCEL = "cancel"
    FILL = "fill"
    MODIFY = "modify"


@dataclass(slots=True)
class BookEvent:
    timestamp: float
    type: EventType
    side: Side | None
    price: int | None
    qty: int
    order_id: int
    is_market: bool = False


@dataclass(slots=True)
class L2Snapshot:
    """Top-of-book snapshot; prices/qtys are best-first arrays padded with 0."""

    timestamp: float
    bid_prices: np.ndarray
    bid_qtys: np.ndarray
    ask_prices: np.ndarray
    ask_qtys: np.ndarray


@dataclass(slots=True)
class QueuePosition:
    price: int
    index: int  # position within the level's visible FIFO
    qty_ahead_visible: int  # visible qty ahead in the queue
    qty_ahead_total: int  # visible + hidden qty ahead (fill priority order)
