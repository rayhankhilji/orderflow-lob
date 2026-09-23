"""Microstructure metrics: pure functions over L2 snapshots and fill tapes.

Unless noted, prices/quantities are in ticks/shares; results are in ticks.
"""

from __future__ import annotations

import numpy as np

from orderflow.book.types import Fill, L2Snapshot, Side


def _best(prices: np.ndarray, qtys: np.ndarray) -> tuple[float, float]:
    """Best (price, qty); (nan, 0) when the side is empty (padded with 0)."""
    if len(prices) == 0 or prices[0] == 0:
        return float("nan"), 0.0
    return float(prices[0]), float(qtys[0])


def l1_imbalance(snap: L2Snapshot) -> float:
    """(bidqty1 - askqty1) / (bidqty1 + askqty1) in [-1, 1]; 0 when a side is empty."""
    _, bq = _best(snap.bid_prices, snap.bid_qtys)
    _, aq = _best(snap.ask_prices, snap.ask_qtys)
    if bq + aq == 0:
        return 0.0
    return (bq - aq) / (bq + aq)


def depth_imbalance(snap: L2Snapshot, levels: int = 5) -> float:
    """Same ratio over cumulative top-``levels`` depth; 0 when book is empty."""
    bq = float(np.sum(snap.bid_qtys[:levels]))
    aq = float(np.sum(snap.ask_qtys[:levels]))
    if bq + aq == 0:
        return 0.0
    return (bq - aq) / (bq + aq)


def ofi(prev: L2Snapshot, nxt: L2Snapshot) -> float:
    """Cont–Kukanov–Stoikov order-flow imbalance from best bid/ask between snapshots.

    e_n = 1{Pb_n >= Pb_{n-1}} qb_n - 1{Pb_n <= Pb_{n-1}} qb_{n-1}
        - 1{Pa_n <= Pa_{n-1}} qa_n + 1{Pa_n >= Pa_{n-1}} qa_{n-1}
    Missing (empty) sides contribute 0.
    """
    pb_p, qb_p = _best(prev.bid_prices, prev.bid_qtys)
    pb_n, qb_n = _best(nxt.bid_prices, nxt.bid_qtys)
    pa_p, qa_p = _best(prev.ask_prices, prev.ask_qtys)
    pa_n, qa_n = _best(nxt.ask_prices, nxt.ask_qtys)
    e = 0.0
    if not np.isnan(pb_p) and not np.isnan(pb_n):
        e += (pb_n >= pb_p) * qb_n - (pb_n <= pb_p) * qb_p
    if not np.isnan(pa_p) and not np.isnan(pa_n):
        e += -(pa_n <= pa_p) * qa_n + (pa_n >= pa_p) * qa_p
    return e


def mid_series(snapshots: list[L2Snapshot]) -> np.ndarray:
    """Mid price per snapshot (nan when either side is empty)."""
    out = np.full(len(snapshots), np.nan)
    for i, s in enumerate(snapshots):
        pb, _ = _best(s.bid_prices, s.bid_qtys)
        pa, _ = _best(s.ask_prices, s.ask_qtys)
        if not np.isnan(pb) and not np.isnan(pa):
            out[i] = (pb + pa) / 2
    return out


def spread_series(snapshots: list[L2Snapshot]) -> np.ndarray:
    """Ask - bid in ticks per snapshot (nan when either side is empty)."""
    out = np.full(len(snapshots), np.nan)
    for i, s in enumerate(snapshots):
        pb, _ = _best(s.bid_prices, s.bid_qtys)
        pa, _ = _best(s.ask_prices, s.ask_qtys)
        if not np.isnan(pb) and not np.isnan(pa):
            out[i] = pa - pb
    return out


def effective_spread(fill: Fill, mid_at_fill: float) -> float:
    """2 * taker_side * (fill_price - mid) in ticks; always >= 0 for a normal fill."""
    return 2 * float(fill.taker_side) * (fill.price - mid_at_fill)


def realized_spread(fill: Fill, mid_after: float) -> float:
    """2 * taker_side * (fill_price - mid(t + h)) in ticks: what the maker kept."""
    return 2 * float(fill.taker_side) * (fill.price - mid_after)


def _mid_at(times: np.ndarray, mids: np.ndarray, t: float) -> float:
    """Last-observation mid at time t (first observation if t precedes all)."""
    idx = int(np.searchsorted(times, t, side="right")) - 1
    return float(mids[max(idx, 0)])


def adverse_selection(
    fills: list[Fill],
    mid_times: np.ndarray,
    mids: np.ndarray,
    horizon_seconds: float,
) -> tuple[np.ndarray, float]:
    """Signed mid move after each fill: taker_side * (mid(t+h) - mid(t)) in ticks.

    Positive means the taker was informed (the maker was adversely selected).
    Returns (per-fill array, mean); mean is nan for an empty fill list.
    """
    times = np.asarray(mid_times, dtype=float)
    mid_vals = np.asarray(mids, dtype=float)
    out = np.empty(len(fills))
    for i, f in enumerate(fills):
        move = _mid_at(times, mid_vals, f.timestamp + horizon_seconds) - _mid_at(
            times, mid_vals, f.timestamp
        )
        out[i] = float(f.taker_side) * move
    return out, float(np.mean(out)) if len(out) else float("nan")


def queue_depletion_time(
    snapshots: list[L2Snapshot],
    side: Side,
    start: int = 0,
    max_events: int | None = None,
) -> int | None:
    """Events until the best ``side`` queue at ``start`` is depleted.

    Depletion = that price level disappears (empty or improved/withdrawn).
    Returns the number of snapshots after ``start``, or None if it survives
    ``max_events`` (default: to the end of the series).
    """
    prices = snapshots[start].bid_prices if side is Side.BUY else snapshots[start].ask_prices
    if len(prices) == 0 or prices[0] == 0:
        return None
    target = prices[0]
    stop = len(snapshots) if max_events is None else min(len(snapshots), start + max_events + 1)
    for i in range(start + 1, stop):
        p = snapshots[i].bid_prices if side is Side.BUY else snapshots[i].ask_prices
        if len(p) == 0 or p[0] != target:
            return i - start
    return None
