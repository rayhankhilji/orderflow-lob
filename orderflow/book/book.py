"""Limit order book with price-time priority matching.

Priority within a price level: all *displayed* liquidity (strict FIFO) fills
before any *hidden* liquidity (strict FIFO). Iceberg orders keep their displayed
peak in the visible deque; when the peak is exhausted the order is re-queued at
the back of the visible deque with a fresh peak of ``min(display_qty, remaining)``
— so a taker sweeping a level consumes an iceberg entirely, peak by peak, while
the reserve stays invisible in ``depth()``.

Fully hidden orders never appear in ``depth()`` (unless ``include_hidden=True``)
and never set ``best_bid``/``best_ask`` — L1 is the *visible* book. They do,
however, fill incoming crossing/market orders at their price once all visible
liquidity at that price is exhausted (real-exchange behaviour).
"""

from __future__ import annotations

from bisect import insort_left

import numpy as np

from orderflow.book.level import PriceLevel
from orderflow.book.types import (
    BookEvent,
    EventType,
    Fill,
    L2Snapshot,
    Order,
    OrderType,
    QueuePosition,
    Side,
    TimeInForce,
)


class LimitOrderBook:
    def __init__(self, tick_size: float = 0.01, record_events: bool = True) -> None:
        self.tick_size = tick_size
        self.record_events = record_events
        self.bids: dict[int, PriceLevel] = {}
        self.asks: dict[int, PriceLevel] = {}
        self._bid_prices: list[int] = []  # ascending; best bid = last
        self._ask_prices: list[int] = []  # ascending; best ask = first
        self.orders: dict[int, Order] = {}
        self._order_level: dict[int, PriceLevel] = {}
        self.events: list[BookEvent] = []
        self.fills: list[Fill] = []
        self.now: float = 0.0

    # ------------------------------------------------------------------ tape
    def _record(self, event: BookEvent) -> None:
        if self.record_events:
            self.events.append(event)

    def _touch(self, t: float) -> None:
        self.now = max(self.now, t)

    # ---------------------------------------------------------------- levels
    def _levels(self, side: Side) -> tuple[dict[int, PriceLevel], list[int]]:
        return (self.bids, self._bid_prices) if side is Side.BUY else (self.asks, self._ask_prices)

    def _add_order(self, order: Order) -> None:
        levels, prices = self._levels(order.side)
        assert order.price is not None
        level = levels.get(order.price)
        if level is None:
            level = PriceLevel(order.price)
            levels[order.price] = level
            insort_left(prices, order.price)
        if order.is_iceberg:
            order.displayed = min(order.display_qty or 0, order.remaining)
            level.visible.append(order)
        elif order.hidden:
            level.hidden.append(order)
        else:
            level.visible.append(order)
        self.orders[order.order_id] = order
        self._order_level[order.order_id] = level

    def _remove_order(self, order: Order) -> None:
        level = self._order_level.pop(order.order_id)
        for dq in (level.visible, level.hidden):
            try:
                dq.remove(order)
                break
            except ValueError:
                continue
        self.orders.pop(order.order_id)
        self._maybe_drop_level(order.side, level)

    def _maybe_drop_level(self, side: Side, level: PriceLevel) -> None:
        if level.n_orders > 0:
            return
        levels, prices = self._levels(side)
        del levels[level.price]
        prices.remove(level.price)

    # -------------------------------------------------------------- matching
    def submit(self, order: Order) -> list[Fill]:
        self._touch(order.timestamp)
        self._record(
            BookEvent(
                timestamp=order.timestamp,
                type=EventType.SUBMIT,
                side=order.side,
                price=order.price,
                qty=order.qty,
                order_id=order.order_id,
                is_market=order.order_type is OrderType.MARKET,
            )
        )
        fills = self._match(order)
        self.fills.extend(fills)
        if order.remaining > 0 and self._rests(order):
            self._add_order(order)
        return fills

    def _rests(self, order: Order) -> bool:
        return order.order_type is OrderType.LIMIT and order.tif is TimeInForce.GTC

    def _match(self, taker: Order) -> list[Fill]:
        opp = Side.SELL if taker.side is Side.BUY else Side.BUY
        levels, prices = self._levels(opp)
        fills: list[Fill] = []
        while taker.remaining > 0 and prices:
            best = prices[-1] if opp is Side.BUY else prices[0]
            if taker.order_type is OrderType.LIMIT:
                assert taker.price is not None
                if taker.side is Side.BUY and taker.price < best:
                    break
                if taker.side is Side.SELL and taker.price > best:
                    break
            level = levels[best]
            self._match_level(level, taker, fills)
            if level.n_orders == 0:
                del levels[best]
                prices.remove(best)
        return fills

    def _match_level(self, level: PriceLevel, taker: Order, fills: list[Fill]) -> None:
        while taker.remaining > 0 and level.visible:
            maker = level.visible[0]
            q = min(maker.visible_remaining, taker.remaining)
            maker.remaining -= q
            if maker.is_iceberg:
                assert maker.displayed is not None
                maker.displayed -= q
            taker.remaining -= q
            fills.append(self._mk_fill(level.price, q, maker, taker, maker.hidden))
            if maker.visible_remaining == 0:
                level.visible.popleft()
                if maker.remaining == 0:
                    self.orders.pop(maker.order_id)
                    self._order_level.pop(maker.order_id)
                elif maker.is_iceberg:
                    # refresh: new peak re-queued at the back of the visible queue
                    maker.displayed = min(maker.display_qty or 0, maker.remaining)
                    level.visible.append(maker)
        while taker.remaining > 0 and level.hidden:
            maker = level.hidden[0]
            q = min(maker.remaining, taker.remaining)
            maker.remaining -= q
            taker.remaining -= q
            fills.append(self._mk_fill(level.price, q, maker, taker, maker.hidden))
            if maker.remaining == 0:
                level.hidden.popleft()
                self.orders.pop(maker.order_id)
                self._order_level.pop(maker.order_id)

    def _mk_fill(self, price: int, qty: int, maker: Order, taker: Order, hidden: bool) -> Fill:
        self._record(
            BookEvent(
                timestamp=taker.timestamp,
                type=EventType.FILL,
                side=taker.side,
                price=price,
                qty=qty,
                order_id=taker.order_id,
                is_market=taker.order_type is OrderType.MARKET,
            )
        )
        return Fill(
            timestamp=taker.timestamp,
            price=price,
            qty=qty,
            maker_id=maker.order_id,
            taker_id=taker.order_id,
            taker_side=taker.side,
            maker_hidden=hidden,
        )

    # ---------------------------------------------------------------- cancel
    def cancel(self, order_id: int, qty: int | None = None) -> bool:
        order = self.orders.get(order_id)
        if order is None or order.remaining <= 0:
            return False
        cancel_qty = order.remaining if qty is None else min(qty, order.remaining)
        self._reduce(order, cancel_qty)
        self._record(
            BookEvent(
                timestamp=self.now,
                type=EventType.CANCEL,
                side=order.side,
                price=order.price,
                qty=cancel_qty,
                order_id=order_id,
            )
        )
        return True

    def _reduce(self, order: Order, qty: int) -> None:
        """Reduce ``order.remaining`` by qty, shrinking hidden/reserve first."""
        hidden_part = min(order.hidden_remaining, qty)
        order.remaining -= qty
        if order.is_iceberg and order.displayed is not None:
            order.displayed -= qty - hidden_part
            order.displayed = max(order.displayed, 0)
        if order.remaining == 0:
            self._remove_order(order)
        elif order.is_iceberg and order.visible_remaining == 0:
            # displayed peak cancelled away but reserve remains: re-queue at back
            level = self._order_level[order.order_id]
            level.visible.remove(order)
            order.displayed = min(order.display_qty or 0, order.remaining)
            level.visible.append(order)

    # ---------------------------------------------------------------- modify
    def modify(
        self, order_id: int, new_price: int | None = None, new_qty: int | None = None
    ) -> list[Fill]:
        order = self.orders.get(order_id)
        if order is None or order.remaining <= 0:
            return []
        target_qty = order.remaining if new_qty is None else new_qty
        keeps_priority = (
            new_price is None or new_price == order.price
        ) and target_qty <= order.remaining
        self._record(
            BookEvent(
                timestamp=self.now,
                type=EventType.MODIFY,
                side=order.side,
                price=new_price if new_price is not None else order.price,
                qty=target_qty,
                order_id=order_id,
            )
        )
        if keeps_priority:
            self._reduce(order, order.remaining - target_qty)
            return []
        # lose priority: cancel + resubmit under the same id with a new timestamp
        self._remove_order(order)
        order.price = order.price if new_price is None else new_price
        order.remaining = target_qty
        order.qty = max(order.qty, target_qty)
        order.timestamp = self.now
        if order.is_iceberg:
            order.displayed = None
        fills = self._match(order)
        self.fills.extend(fills)
        if order.remaining > 0 and self._rests(order):
            self._add_order(order)
        return fills

    # ------------------------------------------------------------------- L1
    def _best_visible(self, side: Side) -> int | None:
        """Best price with displayed liquidity, scanning from the best end."""
        levels, prices = self._levels(side)
        it = reversed(prices) if side is Side.BUY else iter(prices)
        for p in it:
            if levels[p].visible_qty > 0:
                return p
        return None

    def best_bid(self) -> int | None:
        return self._best_visible(Side.BUY)

    def best_ask(self) -> int | None:
        return self._best_visible(Side.SELL)

    def mid(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return (b + a) / 2

    def spread(self) -> int | None:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return a - b

    def microprice(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        bq = self.bids[b].visible_qty
        aq = self.asks[a].visible_qty
        if bq + aq == 0:
            return None
        return (a * bq + b * aq) / (bq + aq)

    # ------------------------------------------------------------------- L2
    def depth(
        self, side: Side, levels: int = 10, include_hidden: bool = False
    ) -> list[tuple[int, int]]:
        book, prices = self._levels(side)
        ordered = prices[::-1] if side is Side.BUY else list(prices)
        out = []
        for p in ordered:
            level = book[p]
            q = level.total_qty if include_hidden else level.visible_qty
            if q > 0:
                out.append((p, q))
            if len(out) == levels:
                break
        return out

    def snapshot(self, levels: int = 10) -> L2Snapshot:
        bids = self.depth(Side.BUY, levels)
        asks = self.depth(Side.SELL, levels)
        bp = np.zeros(levels, dtype=np.int64)
        bq = np.zeros(levels, dtype=np.int64)
        ap = np.zeros(levels, dtype=np.int64)
        aq = np.zeros(levels, dtype=np.int64)
        for i, (p, q) in enumerate(bids):
            bp[i], bq[i] = p, q
        for i, (p, q) in enumerate(asks):
            ap[i], aq[i] = p, q
        return L2Snapshot(
            timestamp=self.now, bid_prices=bp, bid_qtys=bq, ask_prices=ap, ask_qtys=aq
        )

    # ------------------------------------------------------------------- L3
    def get_order(self, order_id: int) -> Order | None:
        return self.orders.get(order_id)

    def queue_position(self, order_id: int) -> QueuePosition:
        """Position in fill priority.

        ``index`` is the rank in execution order across both deques (visible
        orders occupy indices 0..n_vis-1; hidden orders follow). ``qty_ahead_*``
        count the qty that must fill before this order at its price.
        """
        order = self.orders.get(order_id)
        if order is None:
            raise KeyError(f"order {order_id} not resting in book")
        level = self._order_level[order_id]
        qty_ahead_visible = 0
        for i, o in enumerate(level.visible):
            if o.order_id == order_id:
                return QueuePosition(
                    price=level.price,
                    index=i,
                    qty_ahead_visible=qty_ahead_visible,
                    qty_ahead_total=qty_ahead_visible,
                )
            qty_ahead_visible += o.visible_remaining
        qty_ahead_hidden = 0
        n_vis = len(level.visible)
        for j, o in enumerate(level.hidden):
            if o.order_id == order_id:
                return QueuePosition(
                    price=level.price,
                    index=n_vis + j,
                    qty_ahead_visible=qty_ahead_visible,
                    qty_ahead_total=qty_ahead_visible + qty_ahead_hidden,
                )
            qty_ahead_hidden += o.remaining
        raise KeyError(f"order {order_id} not resting in book")

    def level(self, side: Side, price: int) -> PriceLevel | None:
        levels, _ = self._levels(side)
        return levels.get(price)
