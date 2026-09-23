"""Zero-intelligence flow (Smith–Farmer–Gillemot–Iori style).

Independent Poisson clocks: limit arrivals per level with rate decaying in
distance from the best, market arrivals, and a per-order cancellation hazard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.registry import register_flow
from orderflow.sim.actions import CancelOrder, FlowAction, SubmitLimit, SubmitMarket


@dataclass
class ZIParams:
    lambda_limit: float = 1.2  # base rate at level 0 (per side)
    lambda_market: float = 0.8  # total market-order rate (split 50/50)
    mu_cancel: float = 0.02  # per-order cancellation hazard
    depth_decay: float = 0.7  # rate at level l is lambda_limit * exp(-depth_decay * l)
    max_depth: int = 10  # levels eligible for limit placement
    p_inside: float = 0.1  # prob. of improving the best by 1 tick when spread > 1
    size_mu: float = 3.0  # lognormal size params
    size_sigma: float = 1.0
    p_hidden: float = 0.0
    p_iceberg: float = 0.0


@register_flow("zero_intelligence")
class ZeroIntelligenceFlow:
    def __init__(self, params: ZIParams | dict | None = None, **kw) -> None:
        if params is None:
            params = ZIParams(**kw)
        elif isinstance(params, dict):
            params = ZIParams(**params)
        self.p = params

    def _size(self, rng: np.random.Generator) -> int:
        return max(1, round(rng.lognormal(self.p.size_mu, self.p.size_sigma)))

    def _visible_orders(self, book: LimitOrderBook) -> list[int]:
        ids = []
        for levels in (book.bids, book.asks):
            for level in levels.values():
                ids.extend(o.order_id for o in level.visible)
        return ids

    def next_event(
        self, book: LimitOrderBook, t: float, rng: np.random.Generator
    ) -> tuple[float, FlowAction]:
        p = self.p
        visible = self._visible_orders(book)
        n_levels = p.max_depth + 1
        level_w = np.exp(-p.depth_decay * np.arange(n_levels))
        lam_lim_side = p.lambda_limit * float(level_w.sum())
        lam_cancel = p.mu_cancel * len(visible)
        total = 2 * lam_lim_side + p.lambda_market + lam_cancel
        if total <= 0:
            return 1.0, SubmitMarket(Side.BUY, 0)  # dead config: idle
        dt = float(rng.exponential(1.0 / total))
        u = rng.uniform(0, total)
        if u < 2 * lam_lim_side:
            side = Side.BUY if u < lam_lim_side else Side.SELL
            if book.spread() and book.spread() > 1 and rng.uniform() < p.p_inside:
                offset = -1
            else:
                offset = int(rng.choice(n_levels, p=level_w / level_w.sum()))
            qty = self._size(rng)
            hidden = rng.uniform() < p.p_hidden
            display = None
            if not hidden and rng.uniform() < p.p_iceberg:
                display = max(1, qty // 5)
            return dt, SubmitLimit(side, offset, qty, hidden=hidden, display_qty=display)
        u -= 2 * lam_lim_side
        if u < p.lambda_market:
            side = Side.BUY if u < p.lambda_market / 2 else Side.SELL
            return dt, SubmitMarket(side, self._size(rng))
        if visible:
            return dt, CancelOrder(int(visible[rng.integers(len(visible))]))
        return dt, SubmitMarket(Side.BUY, 1)  # nothing to cancel: benign market
