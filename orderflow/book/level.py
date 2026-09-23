"""A single price level: FIFO deques of visible and hidden resting orders."""

from __future__ import annotations

from collections import deque

from orderflow.book.types import Order


class PriceLevel:
    __slots__ = ("hidden", "price", "visible")

    def __init__(self, price: int) -> None:
        self.price = price
        self.visible: deque[Order] = deque()
        self.hidden: deque[Order] = deque()

    @property
    def visible_qty(self) -> int:
        return sum(o.visible_remaining for o in self.visible)

    @property
    def hidden_qty(self) -> int:
        # fully hidden orders + iceberg reserves held in the hidden deque
        return sum(o.remaining for o in self.hidden) + sum(o.hidden_remaining for o in self.visible)

    @property
    def total_qty(self) -> int:
        return self.visible_qty + self.hidden_qty

    @property
    def n_orders(self) -> int:
        return len(self.visible) + len(self.hidden)

    def position_of(self, order_id: int) -> tuple[int, int] | None:
        """Return (index, qty_ahead) for an order in either deque, else None.

        Index/qty_ahead are measured within the order's own deque; hidden orders
        always rank behind all visible orders for execution purposes.
        """
        qty = 0
        for i, o in enumerate(self.visible):
            if o.order_id == order_id:
                return i, qty
            qty += o.visible_remaining
        qty = 0
        for i, o in enumerate(self.hidden):
            if o.order_id == order_id:
                return i, qty
            qty += o.remaining
        return None

    def __len__(self) -> int:
        return self.n_orders
