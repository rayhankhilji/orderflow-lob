"""Queue-reactive flow (Huang–Lehalle–Rosenbaum style).

Intensities at each of the first k levels are functions of that level's
visible queue size q (in units of a reference size qref):

    lam_L(q) = lL / (1 + q/qref)   limit arrivals decrease in q
    lam_C(q) = lC * (q/qref)       cancellations increase in q
    lam_M(q) = lM / (1 + q/qref)   market orders at the best decrease in q

When a best queue is totally depleted, with probability ``1 - theta`` the level
is refilled with a fresh limit order of size qref/2 at the old best (the mid
mean-reverts); with probability ``theta`` nothing refills (reference shifted).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.registry import register_flow
from orderflow.sim.actions import CancelRandom, FlowAction, SubmitLimit, SubmitMarket


@dataclass
class QueueReactiveParams:
    k: int = 3  # levels tracked per side
    qref: float = 50.0  # reference queue size
    lL: tuple[float, ...] = (2.5, 1.5, 1.0)  # limit intensity scale per level
    lC: tuple[float, ...] = (0.10, 0.10, 0.10)  # cancel intensity scale per level
    lM: tuple[float, ...] = (0.8, 0.0, 0.0)  # market intensity scale (best level)
    theta: float = 0.5  # P(reference moves | best queue depleted)
    p_inside: float = 0.3  # prob. a level-0 limit improves by 1 tick when spread > 1
    size_mu: float = 3.0
    size_sigma: float = 0.5
    extra_fields: dict = field(default_factory=dict)


@register_flow("queue_reactive")
class QueueReactiveFlow:
    def __init__(self, params: QueueReactiveParams | dict | None = None, **kw) -> None:
        if params is None:
            params = QueueReactiveParams(**kw)
        elif isinstance(params, dict):
            params = QueueReactiveParams(**params)
        self.p = params
        self._prev_best: dict[Side, int | None] = {Side.BUY: None, Side.SELL: None}

    def _queue_sizes(self, book: LimitOrderBook, side: Side) -> list[float]:
        """Visible queue size at each of the first k levels (0 if absent)."""
        d = book.depth(side, levels=self.p.k)
        return [float(q) for _, q in d] + [0.0] * (self.p.k - len(d))

    def _size(self, rng: np.random.Generator) -> int:
        return max(1, round(rng.lognormal(self.p.size_mu, self.p.size_sigma)))

    def next_event(
        self, book: LimitOrderBook, t: float, rng: np.random.Generator
    ) -> tuple[float, FlowAction]:
        p = self.p
        # depletion check: a best level vanished since last call
        for side in (Side.BUY, Side.SELL):
            prev = self._prev_best[side]
            levels, prices = book._levels(side)
            cur = None
            if side is Side.BUY:
                cur = next((x for x in reversed(prices) if levels[x].visible_qty > 0), None)
            else:
                cur = next((x for x in prices if levels[x].visible_qty > 0), None)
            self._prev_best[side] = cur
            if prev is not None and cur is not None and prev != cur:
                depleted_away = (side is Side.BUY and cur < prev) or (
                    side is Side.SELL and cur > prev
                )
                if depleted_away and rng.uniform() > p.theta:
                    # refill the old best level with a fresh qref/2 order
                    offset = -abs(cur - prev)  # negative: re-post at the old best
                    qty = max(1, int(p.qref // 2))
                    return 0.0, SubmitLimit(side, offset, qty)
        # build rate table
        events: list[tuple[float, FlowAction]] = []
        for side in (Side.BUY, Side.SELL):
            qs = self._queue_sizes(book, side)
            for lvl in range(p.k):
                q = qs[lvl]
                lam_l = p.lL[lvl] / (1.0 + q / p.qref)
                if lam_l > 0:
                    off = lvl
                    spread = book.spread()
                    if lvl == 0 and spread and spread > 1 and rng.uniform() < p.p_inside:
                        off = -1
                    events.append((lam_l, SubmitLimit(side, off, self._size(rng))))
                if q > 0:
                    lam_c = p.lC[lvl] * q / p.qref
                    if lam_c > 0:
                        events.append((lam_c, CancelRandom(side, lvl)))
            lam_m = p.lM[0] / (1.0 + qs[0] / p.qref) if p.lM else 0.0
            if lam_m > 0:
                events.append((lam_m, SubmitMarket(side, self._size(rng))))
        total = sum(lam for lam, _ in events)
        if total <= 0:
            return 1.0, SubmitMarket(Side.BUY, 0)
        dt = float(rng.exponential(1.0 / total))
        u = rng.uniform(0, total)
        acc = 0.0
        for lam, action in events:
            acc += lam
            if u <= acc:
                return dt, action
        return dt, events[-1][1]
