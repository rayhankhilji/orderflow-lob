"""Market-impact estimation: Kyle's lambda, square-root law, propagator, AC."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Order, OrderType, Side
from orderflow.sim.simulator import Simulator, seed_book


def kyle_lambda(signed_volume: np.ndarray, price_change: np.ndarray) -> float:
    """OLS slope of price_change on signed_volume (with intercept)."""
    x = np.asarray(signed_volume, dtype=float)
    y = np.asarray(price_change, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[1])


def square_root_law(volumes: np.ndarray, impacts: np.ndarray, adv: float, sigma: float) -> float:
    """Fit I = Y * sigma * sqrt(Q/V) by least squares through the origin."""
    Q = np.asarray(volumes, dtype=float)
    I = np.asarray(impacts, dtype=float)
    x = sigma * np.sqrt(Q / adv)
    return float((x @ I) / (x @ x))


@dataclass
class PropagatorModel:
    """Bouchaud transient-impact: R(l) = sum_{k<=l} G(k) C(l-k)."""

    G: np.ndarray = field(default_factory=lambda: np.empty(0))  # decay kernel
    response: np.ndarray = field(default_factory=lambda: np.empty(0))  # R(l)
    sign_autocorr: np.ndarray = field(default_factory=lambda: np.empty(0))  # C(l)
    decay_exponent: float = np.nan  # G(l) ~ l^-beta

    @classmethod
    def fit(
        cls,
        signs: np.ndarray,
        price_changes: np.ndarray,
        max_lag: int = 50,
    ) -> PropagatorModel:
        signs = np.asarray(signs, dtype=float)
        pc = np.asarray(price_changes, dtype=float)
        n = min(len(signs), len(pc))
        signs, pc = signs[:n], pc[:n]
        C = np.empty(max_lag + 1)
        R = np.empty(max_lag + 1)
        for l in range(max_lag + 1):
            C[l] = float(np.mean(signs[: n - l] * signs[l:])) if l < n else 0.0
            # response: E[sign_t * sum_{k=1..l} dp_{t+k}]
            R[l] = (
                float(np.mean(signs[: n - l] * (np.cumsum(pc)[l:n] - np.cumsum(pc)[: n - l])))
                if 0 < l < n
                else 0.0
            )
            if l == 0:
                R[0] = 0.0
        # deconvolve: R(l) = sum_{k=1}^{l} G(k) C(l-k), C(0)=1
        G = np.zeros(max_lag + 1)
        for l in range(1, max_lag + 1):
            G[l] = R[l] - sum(G[k] * C[l - k] for k in range(1, l))
        # power-law exponent on the positive part of G
        idx = np.arange(1, max_lag + 1)
        pos = G[1:] > 0
        decay = np.nan
        if pos.sum() >= 3:
            lx, ly = np.log(idx[pos]), np.log(G[1:][pos])
            decay = -float(np.polyfit(lx, ly, 1)[0])
        return cls(G=G, response=R, sign_autocorr=C, decay_exponent=decay)


@dataclass
class ACParams:
    sigma: float  # per-sqrt-second mid volatility ($)
    eta: float  # temporary impact coeff: impact_$ = eta * (shares/sec)
    gamma: float  # permanent impact coeff: impact_$ = gamma * shares
    n_obs: int


class _MetaorderParticipant:
    """Sells `total_qty` in equal market slices every `step` seconds."""

    def __init__(self, side: Side, total_qty: int, step: float, start: float = 0.0) -> None:
        self.name = "metaorder"
        self.side = side
        self.step = step
        self.t = start
        self.n_slices = 0
        self.slice_qty = 0
        self.total_qty = total_qty
        self.sent = 0
        self.slice_mids: list[tuple[float, float]] = []  # (slice send time, mid before)

    def _schedule(self, horizon: float) -> None:
        self.n_slices = max(1, int(horizon / self.step))
        self.slice_qty = max(1, round(self.total_qty / self.n_slices))

    def on_event(self, book, event, t):
        return []

    def next_wakeup(self, t: float) -> float | None:
        if self.sent >= self.n_slices:
            return None
        return self.t + (self.sent + 1) * self.step

    def on_time(self, book: LimitOrderBook, t: float):
        if self.sent >= self.n_slices:
            return []
        mid = book.mid()
        qty = min(self.slice_qty, self.total_qty - self.sent * self.slice_qty)
        if qty <= 0 or mid is None:
            self.sent += 1
            return []
        self.slice_mids.append((t, mid))
        self.sent += 1
        return [
            Order(
                order_id=0,
                side=self.side,
                price=None,
                qty=qty,
                remaining=qty,
                timestamp=t,
                order_type=OrderType.MARKET,
            )
        ]


def calibrate_almgren_chriss(
    flow_factory: Callable[[], object],
    seed_kwargs: dict,
    n_runs: int = 3,
    participation_rates: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2),
    horizon: float = 600.0,
    seed: int = 0,
    tick_size: float = 0.01,
    slice_seconds: float = 10.0,
) -> ACParams:
    """Estimate (sigma, eta, gamma) from controlled metaorder experiments.

    1) sigma: std of mid changes ($) per sqrt(second) from plain runs,
       mid sampled every slice_seconds.
    2) For each participation rate r: run a sim where a MetaorderParticipant
       sells r * expected_volume in equal market slices every slice_seconds.
       Temporary impact per slice (mid_before - avg_fill_price, $) is
       regressed through the origin on slice rate v = slice_qty / slice_seconds
       -> eta. Permanent impact (mid_start - mid_end, $) regressed through the
       origin on total executed Q -> gamma.
    """
    rng = np.random.default_rng(seed)
    # --- pass 1: plain runs for sigma and expected volume ---
    sigmas = []
    volumes = []
    for _ in range(n_runs):
        book = LimitOrderBook(tick_size=tick_size)
        s = int(rng.integers(1 << 30))
        r = np.random.default_rng(s)
        seed_book(book, rng=r, **seed_kwargs)
        sim = Simulator(book, flow_factory(), seed=s)
        res = sim.run(horizon)
        ms = res.mid_series
        if len(ms) > 2:
            # resample mid every slice_seconds (last observation)
            grid = np.arange(0, ms[-1, 0], slice_seconds)
            idx = np.searchsorted(ms[:, 0], grid, side="right") - 1
            idx = np.clip(idx, 0, len(ms) - 1)
            mids = ms[idx, 1] * tick_size
            diffs = np.diff(mids)
            if len(diffs):
                sigmas.append(float(np.std(diffs)) / np.sqrt(slice_seconds))
        volumes.append(float(sum(f.qty for f in res.fills)))
    sigma = float(np.mean(sigmas)) if sigmas else np.nan
    expected_volume = float(np.mean(volumes)) if volumes else 0.0

    # --- pass 2: metaorder runs ---
    temp_impacts: list[float] = []
    trade_rates: list[float] = []
    perm_impacts: list[float] = []
    qtys: list[float] = []
    for rate in participation_rates:
        total_q = int(rate * expected_volume)
        if total_q <= 0:
            continue
        book = LimitOrderBook(tick_size=tick_size)
        s = int(rng.integers(1 << 30))
        r = np.random.default_rng(s)
        seed_book(book, rng=r, **seed_kwargs)
        mp = _MetaorderParticipant(Side.SELL, total_q, step=slice_seconds)
        mp._schedule(horizon)
        sim = Simulator(book, flow_factory(), seed=s, participants=[mp])
        res = sim.run(horizon)
        fills = res.participant_fills.get("metaorder", [])
        ms = res.mid_series
        if not fills or len(ms) < 2:
            continue
        # per-slice temporary impact
        mid_start = ms[0, 1]
        for t_s, mid_before in mp.slice_mids:
            w = [(f.price, f.qty) for f in fills if t_s <= f.timestamp < t_s + slice_seconds]
            if not w:
                continue
            vwap = sum(p_ * q_ for p_, q_ in w) / sum(q_ for _, q_ in w)
            temp_impacts.append((mid_before - vwap) * tick_size)
            trade_rates.append(sum(q_ for _, q_ in w) / slice_seconds)
        mid_end = ms[-1, 1]
        perm_impacts.append((mid_start - mid_end) * tick_size)
        qtys.append(float(sum(f.qty for f in fills)))

    def thru_origin(y: list[float], x: list[float]) -> float:
        ya, xa = np.asarray(y), np.asarray(x)
        if len(ya) == 0 or float(xa @ xa) == 0:
            return np.nan
        return float((xa @ ya) / (xa @ xa))

    return ACParams(
        sigma=sigma,
        eta=thru_origin(temp_impacts, trade_rates),
        gamma=thru_origin(perm_impacts, qtys),
        n_obs=len(temp_impacts) + len(perm_impacts),
    )
