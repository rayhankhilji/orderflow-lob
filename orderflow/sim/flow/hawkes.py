"""Multivariate Hawkes flow with exponential kernels (Ogata thinning).

Event types: 0 buy_limit, 1 sell_limit, 2 buy_market, 3 sell_market,
4 buy_cancel, 5 sell_cancel. Intensity

    lambda_i(t) = mu_i + sum_j A_ij(t),
    A_ij(t) = sum_{t_k^j < t} alpha_ij exp(-beta_ij (t - t_k^j))

maintained incrementally: A_ij decays exponentially between events and jumps
by alpha_ij on each type-j event. Spectral radius of alpha/beta must be < 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.registry import register_flow
from orderflow.sim.actions import CancelRandom, FlowAction, SubmitLimit, SubmitMarket

BUY_LIMIT, SELL_LIMIT, BUY_MARKET, SELL_MARKET, BUY_CANCEL, SELL_CANCEL = range(6)


@dataclass
class HawkesParams:
    mu: np.ndarray  # (6,)
    alpha: np.ndarray  # (6,6) excitation alpha_ij (i excited by j)
    beta: np.ndarray | float  # (6,6) or scalar shared decay

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float)
        self.alpha = np.asarray(self.alpha, dtype=float)
        beta = np.asarray(self.beta, dtype=float)
        n = len(self.mu)
        self.beta = np.full((n, n), float(beta)) if beta.ndim == 0 else beta

    def branching_ratio(self) -> float:
        kernel = self.alpha / self.beta
        return float(np.max(np.abs(np.linalg.eigvals(kernel))))


def simulate_hawkes(params: HawkesParams, T: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Pure Ogata thinning: returns per-type sorted event times (no book)."""
    n = 6
    A = np.zeros((n, n))
    times: list[list[float]] = [[] for _ in range(n)]
    t = 0.0
    while True:
        lam_bar = float((params.mu + A.sum(axis=1)).sum())
        if lam_bar <= 0:
            break
        t_prop = t + rng.exponential(1.0 / lam_bar)
        if t_prop > T:
            break
        A *= np.exp(-params.beta * (t_prop - t))
        t = t_prop
        lam_now = params.mu + A.sum(axis=1)
        if rng.uniform() * lam_bar <= lam_now.sum():
            i = int(rng.choice(n, p=lam_now / lam_now.sum()))
            A[:, i] += params.alpha[:, i]
            times[i].append(t)
        # rejected proposals just advance the clock (A already decayed)
    return [np.asarray(ts, dtype=float) for ts in times]


@register_flow("hawkes")
class HawkesFlow:
    def __init__(self, params: HawkesParams | dict | None = None, **kw) -> None:
        cfg = dict(params) if isinstance(params, dict) else {}
        cfg.update(kw)
        self.depth_decay = cfg.pop("depth_decay", 0.7)
        self.max_depth = cfg.pop("max_depth", 10)
        self.size_mu = cfg.pop("size_mu", 3.0)
        self.size_sigma = cfg.pop("size_sigma", 1.0)
        self.p_hidden = cfg.pop("p_hidden", 0.0)
        self.p_iceberg = cfg.pop("p_iceberg", 0.0)
        self.p_inside = cfg.pop("p_inside", 0.2)
        if params is None or isinstance(params, dict):
            params = HawkesParams(**cfg)
        assert params.branching_ratio() < 1, (
            f"Hawkes branching ratio {params.branching_ratio():.3f} >= 1"
        )
        self.params = params
        self._A = np.zeros((6, 6))
        self._t = 0.0

    def _draw_dt(self, rng: np.random.Generator) -> tuple[float, int]:
        """Ogata's modified thinning from internal clock; returns (dt, event_type)."""
        p = self.params
        t = self._t
        A = self._A
        while True:
            lam = p.mu + A.sum(axis=1)
            lam_bar = float(lam.sum())
            t += rng.exponential(1.0 / lam_bar)
            # decay excitation to proposal time
            dt_prop = t - self._t
            A *= np.exp(-p.beta * dt_prop)
            self._t = t
            lam_now = p.mu + A.sum(axis=1)
            if rng.uniform() * lam_bar <= lam_now.sum():
                i = int(rng.choice(6, p=lam_now / lam_now.sum()))
                A[:, i] += p.alpha[:, i]
                return t, i
            # rejected: continue thinning (A already decayed, _t advanced)

    def next_event(
        self, book: LimitOrderBook, t: float, rng: np.random.Generator
    ) -> tuple[float, FlowAction]:
        # fast-forward internal clock (decay only, no events) to sim time
        self._A *= np.exp(-self.params.beta * max(t - self._t, 0.0))
        self._t = max(t, self._t)
        t_ev, i = self._draw_dt(rng)
        return max(t_ev - t, 0.0), self._to_action(i, rng, book)

    def _size(self, rng: np.random.Generator) -> int:
        return max(1, round(rng.lognormal(self.size_mu, self.size_sigma)))

    def _offset(self, rng: np.random.Generator) -> int:
        w = np.exp(-self.depth_decay * np.arange(self.max_depth + 1))
        return int(rng.choice(self.max_depth + 1, p=w / w.sum()))

    def _to_action(self, i: int, rng: np.random.Generator, book: LimitOrderBook) -> FlowAction:
        if i in (BUY_LIMIT, SELL_LIMIT):
            side = Side.BUY if i == BUY_LIMIT else Side.SELL
            qty = self._size(rng)
            hidden = rng.uniform() < self.p_hidden
            display = max(1, qty // 5) if (not hidden and rng.uniform() < self.p_iceberg) else None
            offset = self._offset(rng)
            spread = book.spread()
            if spread and spread > 1 and rng.uniform() < self.p_inside:
                offset = -1
            return SubmitLimit(side, offset, qty, hidden=hidden, display_qty=display)
        if i in (BUY_MARKET, SELL_MARKET):
            return SubmitMarket(Side.BUY if i == BUY_MARKET else Side.SELL, self._size(rng))
        side = Side.BUY if i == BUY_CANCEL else Side.SELL
        return CancelRandom(side, int(rng.integers(0, 3)))
