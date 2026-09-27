"""Adversarial participants used to stress execution algos.

Each adversary watches the tape for an execution footprint — market-order
submits of ``footprint_qty`` or larger — and attacks the algo while the
footprint is active. Adversaries are scored by how much they raise each
algo's implementation shortfall (eval/execution_bench runs algo-alone vs
algo+adversary on identical seeds).
"""

from __future__ import annotations

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import BookEvent, EventType, Order, OrderType, Side
from orderflow.sim.actions import CancelOrder, CancelRandom


def _market(ev: BookEvent) -> bool:
    return ev.type is EventType.SUBMIT and ev.is_market


class _Decay:
    """Footprint detection with a decaying activity window.

    ``mine`` holds the adversary's own order ids so its own market orders
    don't retrigger detection (which would recurse the digest loop forever).
    """

    def __init__(self, footprint_qty: int, decay: float, mine: set[int]) -> None:
        self.footprint_qty = footprint_qty
        self.decay = decay
        self.mine = mine
        self.active_until = -np.inf
        self.side: Side | None = None

    def see(self, ev: BookEvent, t: float) -> bool:
        if ev.order_id in self.mine:
            return False
        if _market(ev) and ev.qty >= self.footprint_qty:
            self.active_until = t + self.decay
            self.side = ev.side
            return True
        return False

    def active(self, t: float) -> bool:
        return t <= self.active_until


class Spoofer:
    """Posts large fake depth on the side the execution consumes, pulled just
    before it can be hit. Inflates displayed depth to distort depth-imbalance
    signals and attract contra flow that lifts the price mid-execution."""

    def __init__(
        self,
        footprint_qty: int = 200,
        spoof_qty: int = 800,
        offset_ticks: int = 3,
        lifetime: float = 1.5,
        decay: float = 3.0,
        name: str = "adv_spoof",
    ) -> None:
        self.name = name
        self._my_ids: set[int] = set()
        self.det = _Decay(footprint_qty, decay, self._my_ids)
        self.spoof_qty = spoof_qty
        self.offset_ticks = offset_ticks
        self.lifetime = lifetime
        self._live: list[tuple[float, int]] = []  # (cancel_at, order_id)
        self._next_id = 1 << 30

    def on_event(self, book: LimitOrderBook, ev: BookEvent, t: float):
        if not self.det.see(ev, t) or self.det.side is None:
            return []
        side = self.det.side
        best = book.best_bid() if side is Side.BUY else book.best_ask()
        if best is None:
            return []
        price = best - self.offset_ticks if side is Side.BUY else best + self.offset_ticks
        oid = self._next_id
        self._next_id += 1
        self._my_ids.add(oid)
        self._live.append((t + self.lifetime, oid))
        return [
            Order(
                order_id=oid,
                side=side,
                price=int(price),
                qty=self.spoof_qty,
                remaining=self.spoof_qty,
                timestamp=t,
                order_type=OrderType.LIMIT,
            )
        ]

    def next_wakeup(self, t: float) -> float | None:
        live = [w for w, _ in self._live]
        return min(live) if live else None

    def on_time(self, book: LimitOrderBook, t: float):
        out = []
        keep = []
        for w, oid in self._live:
            if w <= t:
                out.append(CancelOrder(oid))
            else:
                keep.append((w, oid))
        self._live = keep
        return out


class MomentumIgniter:
    """Fires same-direction market orders while the footprint is active —
    pushes the mid against the execution (sell exec -> it sells too)."""

    def __init__(
        self,
        footprint_qty: int = 200,
        ignite_qty: int = 150,
        interval: float = 1.0,
        decay: float = 4.0,
        name: str = "adv_ignite",
    ) -> None:
        self.name = name
        self._my_ids: set[int] = set()
        self.det = _Decay(footprint_qty, decay, self._my_ids)
        self.ignite_qty = ignite_qty
        self.interval = interval
        self._next_fire = 0.0
        self._next_id = 1 << 31

    def on_event(self, book: LimitOrderBook, ev: BookEvent, t: float):
        if self.det.see(ev, t):
            self._next_fire = t + self.interval
        return []

    def next_wakeup(self, t: float) -> float | None:
        return self._next_fire if self.det.active(t) else None

    def on_time(self, book: LimitOrderBook, t: float):
        self._next_fire = t + self.interval
        if not self.det.active(t) or self.det.side is None:
            return []
        oid = self._next_id
        self._next_id += 1
        self._my_ids.add(oid)
        return [
            Order(
                order_id=oid,
                side=self.det.side,
                price=None,
                qty=self.ignite_qty,
                remaining=self.ignite_qty,
                timestamp=t,
                order_type=OrderType.MARKET,
            )
        ]


class LiquidityWithdrawer:
    """Cancels resting depth on the side the execution needs. For a sell
    metaorder it cancels bids at the touch, thinning liquidity just when the
    slices arrive."""

    def __init__(
        self,
        footprint_qty: int = 200,
        cancels_per_event: int = 3,
        level_offset: int = 0,
        decay: float = 2.0,
        name: str = "adv_withdraw",
    ) -> None:
        self.name = name
        self._my_ids: set[int] = set()
        self.det = _Decay(footprint_qty, decay, self._my_ids)
        self.n = cancels_per_event
        self.level_offset = level_offset

    def on_event(self, book: LimitOrderBook, ev: BookEvent, t: float):
        if not self.det.see(ev, t) or self.det.side is None:
            return []
        contra = Side.BUY if self.det.side is Side.SELL else Side.SELL
        return [
            CancelRandom(contra, level_offset=self.level_offset + i)
            for i in range(self.n)
        ]

    def next_wakeup(self, t: float) -> float | None:
        return None

    def on_time(self, book: LimitOrderBook, t: float):
        return []


class Frontrunner:
    """Trades ahead of the detected slices: when the footprint fires, it sends
    its own smaller same-direction market order first, then lets the exec
    slice push the price it just took. Also unwinds after the burst."""

    def __init__(
        self,
        footprint_qty: int = 200,
        front_qty: int = 100,
        unwind_after: float = 8.0,
        unwind_qty: int = 100,
        decay: float = 3.0,
        name: str = "adv_front",
    ) -> None:
        self.name = name
        self._my_ids: set[int] = set()
        self.det = _Decay(footprint_qty, decay, self._my_ids)
        self.front_qty = front_qty
        self.unwind_after = unwind_after
        self.unwind_qty = unwind_qty
        self._unwind_at: float | None = None
        self._next_id = 1 << 32

    def _mo(self, side: Side, qty: int, t: float) -> Order:
        self._next_id += 1
        self._my_ids.add(self._next_id)
        return Order(
            order_id=self._next_id,
            side=side,
            price=None,
            qty=qty,
            remaining=qty,
            timestamp=t,
            order_type=OrderType.MARKET,
        )

    def on_event(self, book: LimitOrderBook, ev: BookEvent, t: float):
        if not self.det.see(ev, t) or self.det.side is None:
            return []
        self._unwind_at = t + self.unwind_after
        return [self._mo(self.det.side, self.front_qty, t)]

    def next_wakeup(self, t: float) -> float | None:
        return self._unwind_at

    def on_time(self, book: LimitOrderBook, t: float):
        self._unwind_at = None
        if self.det.side is None:
            return []
        return [self._mo(Side(-self.det.side), self.unwind_qty, t)]


ADVERSARIES = {
    "spoofer": Spoofer,
    "igniter": MomentumIgniter,
    "withdrawer": LiquidityWithdrawer,
    "frontrunner": Frontrunner,
}
