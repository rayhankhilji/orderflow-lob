"""Event-driven simulator: a book, a stochastic flow model, and participants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import (
    BookEvent,
    EventType,
    Fill,
    L2Snapshot,
    Order,
    OrderType,
    Side,
    TimeInForce,
)
from orderflow.sim.actions import CancelOrder, CancelRandom, FlowAction, SubmitLimit, SubmitMarket

# Hawkes-compatible event-type names, indexed by (EventType-ish, side)
EVENT_TYPES = (
    "buy_limit",
    "sell_limit",
    "buy_market",
    "sell_market",
    "buy_cancel",
    "sell_cancel",
)


class FlowModel(Protocol):
    def next_event(
        self, book: LimitOrderBook, t: float, rng: np.random.Generator
    ) -> tuple[float, FlowAction]:
        """Return (dt until next event, action)."""
        ...


class Participant(Protocol):
    name: str

    def on_event(
        self, book: LimitOrderBook, event: BookEvent, t: float
    ) -> list[Order | CancelOrder]:
        """React to a book event; returned orders execute immediately."""
        ...

    def on_time(self, book: LimitOrderBook, t: float) -> list[Order | CancelOrder]:
        """Called when the sim clock reaches a wakeup returned by next_wakeup."""
        ...

    def next_wakeup(self, t: float) -> float | None:
        """Next time at which on_time should fire, or None."""
        ...


@dataclass
class SimResult:
    snapshots: list[L2Snapshot]
    events: list[BookEvent]
    fills: list[Fill]
    mid_series: np.ndarray  # (n, 2) rows of (t, mid) at each mid change
    participant_fills: dict[str, list[Fill]]
    event_times_by_type: dict[str, np.ndarray]
    horizon: float = 0.0
    # flow-originated events only (first BookEvent of each flow action), and —
    # when record_state_every_event=True — the L2 state seen just BEFORE each,
    # as rows [bid_prices(L), bid_qtys(L), ask_prices(L), ask_qtys(L), t]
    flow_events: list[BookEvent] = field(default_factory=list)
    event_states: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


def seed_book(
    book: LimitOrderBook,
    mid_ticks: int,
    levels: int,
    qty_per_level: int,
    rng: np.random.Generator,
    jitter: bool = True,
) -> list[int]:
    """Populate both sides so simulation never starts empty; 1-tick spread."""
    ids = []
    for i in range(levels):
        for side, price in ((Side.BUY, mid_ticks - i), (Side.SELL, mid_ticks + 1 + i)):
            q = qty_per_level
            if jitter:
                q = max(
                    1,
                    qty_per_level + int(rng.integers(-qty_per_level // 4, qty_per_level // 4 + 1)),
                )
            oid = -(len(ids) + 1)  # negative ids: seeded liquidity
            book.submit(
                Order(
                    order_id=oid,
                    side=side,
                    price=price,
                    qty=q,
                    remaining=q,
                    timestamp=book.now,
                )
            )
            ids.append(oid)
    return ids


class Simulator:
    """Advances a LimitOrderBook under a FlowModel plus timed/event Participants."""

    def __init__(
        self,
        book: LimitOrderBook,
        flow: FlowModel,
        seed: int,
        participants: tuple[Participant, ...] | list[Participant] = (),
        record_state_every_event: bool = False,
        state_levels: int = 5,
    ) -> None:
        self.book = book
        self.flow = flow
        self.rng = np.random.default_rng(seed)
        self.participants = list(participants)
        self.t = 0.0
        self._id = 0
        self._last_mid: float = float(book.mid() or 0.0)
        self._pending: tuple[float, FlowAction] | None = None
        self._owners: dict[int, str] = {}
        self._event_times: dict[str, list[float]] = {k: [] for k in EVENT_TYPES}
        self._fill_cursor = 0
        self._record_state = record_state_every_event
        self._state_levels = state_levels
        self.flow_events: list[BookEvent] = []
        self._event_states: list[np.ndarray] = []
        self._mid_series: list[tuple[float, float]] = []
        if self._last_mid:
            self._mid_series.append((0.0, self._last_mid))
        self.participant_fills: dict[str, list[Fill]] = {p.name: [] for p in self.participants}

    def next_id(self) -> int:
        self._id += 1
        return self._id

    # ------------------------------------------------------------- applying
    def _anchor(self, side: Side) -> float:
        """Reference price for SubmitLimit offsets."""
        best = self.book.best_bid() if side is Side.BUY else self.book.best_ask()
        if best is not None:
            return float(best)
        # same side empty: anchor off the other side or last mid, 1 tick away
        other = self.book.best_ask() if side is Side.BUY else self.book.best_bid()
        ref = float(other) if other is not None else self._last_mid
        return ref - 1.0 if side is Side.BUY else ref + 1.0

    def _resolve_price(self, side: Side, offset: int) -> int:
        anchor = self._anchor(side)
        return round(anchor - offset) if side is Side.BUY else round(anchor + offset)

    def _apply_action(self, action: FlowAction, t: float, owner: str = "flow") -> list[BookEvent]:
        n0 = len(self.book.events)
        if isinstance(action, SubmitLimit):
            order = Order(
                order_id=self.next_id(),
                side=action.side,
                price=self._resolve_price(action.side, action.price_offset_ticks),
                qty=action.qty,
                remaining=action.qty,
                timestamp=t,
                order_type=OrderType.LIMIT,
                tif=TimeInForce.GTC,
                hidden=action.hidden,
                display_qty=action.display_qty,
                owner=owner,
            )
            self._owners[order.order_id] = owner
            self.book.submit(order)
        elif isinstance(action, SubmitMarket):
            order = Order(
                order_id=self.next_id(),
                side=action.side,
                price=None,
                qty=action.qty,
                remaining=action.qty,
                timestamp=t,
                order_type=OrderType.MARKET,
                owner=owner,
            )
            self._owners[order.order_id] = owner
            self.book.submit(order)
        elif isinstance(action, CancelRandom):
            depth = self.book.depth(action.side, levels=action.level_offset + 1)
            if len(depth) > action.level_offset:
                level = self.book.level(action.side, depth[action.level_offset][0])
                if level is not None and level.visible:
                    victim = list(level.visible)[self.rng.integers(len(level.visible))]
                    self.book.cancel(victim.order_id)
        elif isinstance(action, CancelOrder):
            self.book.cancel(action.order_id)
        else:  # pragma: no cover - defensive
            raise TypeError(f"unknown action {action!r}")
        return self.book.events[n0:]

    def _apply_participant_req(
        self, req: Order | CancelOrder, t: float, owner: str
    ) -> list[BookEvent]:
        n0 = len(self.book.events)
        if isinstance(req, Order):
            req.owner = owner
            req.timestamp = t
            if req.order_id == 0:
                req.order_id = self.next_id()
            self._owners[req.order_id] = owner
            self.book.submit(req)
        elif isinstance(req, CancelOrder):
            self.book.cancel(req.order_id)
        elif isinstance(req, CancelRandom):
            return self._apply_action(req, t, owner)
        else:  # pragma: no cover
            raise TypeError(f"unknown participant request {req!r}")
        return self.book.events[n0:]

    def _classify(self, ev: BookEvent) -> None:
        if ev.type is EventType.SUBMIT:
            key = ("buy" if ev.side is Side.BUY else "sell") + (
                "_market" if ev.is_market else "_limit"
            )
            self._event_times[key].append(ev.timestamp)
        elif ev.type is EventType.CANCEL:
            key = ("buy" if ev.side is Side.BUY else "sell") + "_cancel"
            self._event_times[key].append(ev.timestamp)

    def _digest(self, events: list[BookEvent], t: float) -> None:
        """Record event times, mid changes, participant fills; notify participants."""
        for ev in events:
            self._classify(ev)
        # participant fills: any fill where maker or taker belongs to a participant
        n_new = len(self.book.fills)
        for f in self.book.fills[self._fill_cursor :]:
            owners = {self._owners.get(f.maker_id), self._owners.get(f.taker_id)}
            for name in owners:
                if name and name != "flow" and name in self.participant_fills:
                    self.participant_fills[name].append(f)
        self._fill_cursor = n_new
        mid = self.book.mid()
        if mid is not None and mid != self._last_mid:
            self._mid_series.append((t, mid))
            self._last_mid = mid
        # notify participants of these events; apply their requests immediately
        for ev in events:
            for p in self.participants:
                for req in p.on_event(self.book, ev, t) or ():
                    self._digest(self._apply_participant_req(req, t, p.name), t)

    # ----------------------------------------------------------------- loop
    def step(self) -> BookEvent:
        """Advance one step (flow event or participant wakeup) and return its event."""
        if self._pending is None:
            dt, action = self.flow.next_event(self.book, self.t, self.rng)
            self._pending = (self.t + max(dt, 0.0), action)
        t_event, action = self._pending
        # a participant wakeup may precede the pending flow event
        wake = None
        for p in self.participants:
            w = p.next_wakeup(self.t)
            if w is not None and (wake is None or w < wake):
                wake = w
        if wake is not None and wake <= t_event:
            self.t = max(wake, self.t)
            events: list[BookEvent] = []
            for p in self.participants:
                w = p.next_wakeup(self.t)
                if w is not None and w <= self.t:
                    for req in p.on_time(self.book, self.t) or ():
                        events += self._apply_participant_req(req, self.t, p.name)
            self._digest(events, self.t)
            return events[0] if events else BookEvent(self.t, EventType.MODIFY, None, None, 0, -1)
        self._pending = None
        self.t = t_event
        if self._record_state:
            snap = self.book.snapshot(self._state_levels)
            row = np.concatenate(
                [snap.bid_prices, snap.bid_qtys, snap.ask_prices, snap.ask_qtys, [self.t]]
            )
        else:
            row = None
        events = self._apply_action(action, self.t)
        if events:
            self.flow_events.append(events[0])
            if row is not None:
                self._event_states.append(row)
        self._digest(events, self.t)
        return events[0] if events else BookEvent(self.t, EventType.MODIFY, None, None, 0, -1)

    def run(self, horizon: float, snapshot_every: float | None = None) -> SimResult:
        snapshots: list[L2Snapshot] = []
        next_snap = snapshot_every if snapshot_every else np.inf
        n_events0 = len(self.book.events)
        n_fills0 = len(self.book.fills)
        n_flow0 = len(self.flow_events)
        while self.t < horizon:
            self.step()
            if snapshot_every and self.t >= next_snap:
                snapshots.append(self.book.snapshot(levels=10))
                while next_snap <= self.t:
                    next_snap += snapshot_every
        # trailing snapshot state not needed; final digest already done in step
        events = self.book.events[n_events0:]
        fills = self.book.fills[n_fills0:]
        return SimResult(
            snapshots=snapshots,
            events=events,
            fills=fills,
            mid_series=np.array(self._mid_series, dtype=float).reshape(-1, 2),
            participant_fills=self.participant_fills,
            event_times_by_type={k: np.array(v, dtype=float) for k, v in self._event_times.items()},
            horizon=horizon,
            flow_events=self.flow_events[n_flow0:],
            event_states=np.asarray(self._event_states[n_flow0:], dtype=np.float64).reshape(
                -1, 4 * self._state_levels + 1
            ),
        )
