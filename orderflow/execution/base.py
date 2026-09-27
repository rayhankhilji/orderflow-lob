"""ExecutionAlgo protocol and the Participant adapter that drives it.

An :class:`ExecutionAlgo` is deliberately dumb: given the book and its own fill
state at a decision time, it returns the child orders it wants working now.
:class:`AlgoParticipant` handles everything else — the decision clock, limit
re-quoting between steps, fill bookkeeping and the hard deadline. Child orders
get ids from a private high range so the adapter always knows which fills are
its own without relying on participant_fills attribution.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import BookEvent, Fill, Order, OrderType, Side
from orderflow.execution.types import ChildOrder, ExecState, ExecutionTask
from orderflow.sim.actions import CancelOrder

_BASE_ID = 1 << 40  # algo order ids; well above sim.next_id() and seed ids


class ExecutionAlgo(Protocol):
    name: str

    def reset(
        self, task: ExecutionTask, book: LimitOrderBook, rng: np.random.Generator
    ) -> None: ...

    def decide(
        self, t: float, book: LimitOrderBook, state: ExecState
    ) -> list[ChildOrder]: ...


class AlgoParticipant:
    """Drives an ExecutionAlgo inside a Simulator as a timed Participant."""

    def __init__(self, algo: ExecutionAlgo, task: ExecutionTask) -> None:
        self.algo = algo
        self.task = task
        self.name = algo.name
        # decision wakeups strictly inside the horizon, then the deadline
        n_inner = 0
        while (n_inner + 1) * task.step < task.horizon - 1e-9:
            n_inner += 1
        self._times = [task.start_time + (j + 1) * task.step for j in range(n_inner)]
        self._times.append(task.start_time + task.horizon)
        self._n_done = 0
        self._next_id = _BASE_ID
        self._my_ids: set[int] = set()
        self._open: dict[int, Order] = {}
        self._fills: list[Fill] = []
        self._fill_cursor = 0
        self._arrival_mid: float | None = None
        self.remaining = task.total_qty

    # ------------------------------------------------------------ helpers
    def _alloc_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _collect_fills(self, book: LimitOrderBook) -> None:
        """Advance over book.fills; keep the ones whose maker/taker is ours."""
        for f in book.fills[self._fill_cursor :]:
            if f.maker_id in self._my_ids or f.taker_id in self._my_ids:
                self._fills.append(f)
        self._fill_cursor = len(book.fills)
        self.remaining = self.task.total_qty - sum(f.qty for f in self._fills)

    def _resolve_limit_price(self, book: LimitOrderBook, side: Side, offset: int) -> int:
        """Same anchoring as Simulator._resolve_price, without sim internals."""
        best = book.best_bid() if side is Side.BUY else book.best_ask()
        if best is None:
            other = book.best_ask() if side is Side.BUY else book.best_bid()
            mid = book.mid()
            ref = float(other) if other is not None else float(mid or 0.0)
            anchor = ref - 1.0 if side is Side.BUY else ref + 1.0
        else:
            anchor = float(best)
        price = anchor - offset if side is Side.BUY else anchor + offset
        return round(price)

    def _to_order(self, child: ChildOrder, t: float, book: LimitOrderBook) -> Order:
        qty = max(0, min(int(child.qty), self.remaining))
        oid = self._alloc_id()
        self._my_ids.add(oid)
        if child.kind == "limit":
            return Order(
                order_id=oid,
                side=child.side,
                price=self._resolve_limit_price(
                    book, child.side, child.price_offset_ticks
                ),
                qty=qty,
                remaining=qty,
                timestamp=t,
                order_type=OrderType.LIMIT,
            )
        return Order(
            order_id=oid,
            side=child.side,
            price=None,
            qty=qty,
            remaining=qty,
            timestamp=t,
            order_type=OrderType.MARKET,
        )

    def _liquidate(self, t: float, book: LimitOrderBook) -> list[Order | CancelOrder]:
        """Cancel open children and market-order everything left."""
        reqs: list[Order | CancelOrder] = [
            CancelOrder(oid) for oid in list(self._open)
        ]
        if self.remaining > 0:
            reqs.append(
                self._to_order(
                    ChildOrder(side=self.task.side, qty=self.remaining, kind="market"),
                    t,
                    book,
                )
            )
        return reqs

    # ------------------------------------------------------ Participant API
    def on_event(self, book: LimitOrderBook, event: BookEvent, t: float):
        self._collect_fills(book)
        return []

    def next_wakeup(self, t: float) -> float | None:
        if self._n_done >= len(self._times):
            return None
        return self._times[self._n_done]

    def on_time(self, book: LimitOrderBook, t: float):
        self._collect_fills(book)
        if self._arrival_mid is None:
            mid = book.mid()
            if mid is not None:
                self._arrival_mid = float(mid)
        self._n_done += 1
        if self.remaining <= 0:
            return []

        deadline = self.task.start_time + self.task.horizon
        is_deadline = t >= deadline - 1e-9
        # replace semantics: every step re-quotes the whole working set
        reqs: list[Order | CancelOrder] = [
            CancelOrder(oid) for oid in list(self._open)
        ]
        self._open.clear()
        if is_deadline:
            return reqs + self._liquidate(t, book)

        state = ExecState(
            remaining_qty=self.remaining,
            elapsed=t - self.task.start_time,
            fills=list(self._fills),
            arrival_mid=self._arrival_mid,
        )
        for child in self.algo.decide(t, book, state) or []:
            if child.qty <= 0 or child.side is not self.task.side:
                continue
            order = self._to_order(child, t, book)
            reqs.append(order)
            if order.order_type is OrderType.LIMIT:
                self._open[order.order_id] = order
        return reqs
