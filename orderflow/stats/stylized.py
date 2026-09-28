"""Stylised-fact checks on simulated order flow.

Real equity markets show, among other things: fat-tailed returns, long-memory
volatility clustering, positive long-lag autocorrelation of trade signs, and a
spread that widens as book imbalance collapses. These functions measure each
of those on a SimResult so the flow models can be judged against them (see
research/microstructure/REPORT.md).
"""

from __future__ import annotations

import numpy as np

from orderflow.book.metrics import l1_imbalance, spread_series
from orderflow.book.types import L2Snapshot
from orderflow.sim.simulator import SimResult


def resample_mid(mid_ser: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Last-observation resample of an irregular (t, mid) series to a grid."""
    ms = np.asarray(mid_ser, dtype=float).reshape(-1, 2)
    if len(ms) < 2:
        return np.empty(0)
    grid = np.arange(0.0, ms[-1, 0], dt)
    idx = np.clip(np.searchsorted(ms[:, 0], grid, side="right") - 1, 0, len(ms) - 1)
    return ms[idx, 1]


def return_moments(returns: np.ndarray) -> dict[str, float]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 4:
        return {"mean": np.nan, "std": np.nan, "skew": np.nan, "excess_kurtosis": np.nan}
    m, s = float(np.mean(r)), float(np.std(r))
    z = (r - m) / (s + 1e-15)
    return {
        "mean": m,
        "std": s,
        "skew": float(np.mean(z**3)),
        "excess_kurtosis": float(np.mean(z**4) - 3.0),
    }


def acf(x: np.ndarray, lags: int) -> np.ndarray:
    """Sample autocorrelation rho(1..lags) of a stationary series."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(x @ x)
    out = np.full(lags, np.nan)
    if denom <= 0:
        return out
    n = len(x)
    for k in range(1, lags + 1):
        out[k - 1] = float(x[k:] @ x[:-k]) / denom if k < n else np.nan
    return out


def trade_signs(res: SimResult) -> np.ndarray:
    """+1/-1 per market-order-driven fill (the taker's direction)."""
    return np.array([float(f.taker_side) for f in res.fills])


def spread_imbalance_stats(
    snapshots: list[L2Snapshot],
) -> dict[str, float]:
    """Correlation of spread with |L1 imbalance| and their means.

    Microstructure lore: wider spreads coincide with weaker (near-zero,
    contested) imbalance; a strong one-sided book pulls the touch tight.
    """
    spreads = spread_series(snapshots)
    imb = np.array([l1_imbalance(s) for s in snapshots])
    good = np.isfinite(spreads) & np.isfinite(imb)
    if good.sum() < 4:
        return {"corr_spread_absimb": np.nan, "mean_spread": np.nan,
                "mean_abs_imb": np.nan}
    return {
        "corr_spread_absimb": float(
            np.corrcoef(spreads[good], np.abs(imb[good]))[0, 1]
        ),
        "mean_spread": float(np.mean(spreads[good])),
        "mean_abs_imb": float(np.mean(np.abs(imb[good]))),
    }


def stylized_facts(res: SimResult, ret_dt: float = 1.0, lags: int = 50) -> dict:
    """One-call summary used by REPORT.md and the eval notes."""
    mids = resample_mid(res.mid_series, ret_dt)
    rets = np.diff(mids)
    out = {
        "moments": return_moments(rets),
        "absret_acf": acf(np.abs(rets), lags).tolist(),
        "sign_acf": acf(trade_signs(res), lags).tolist(),
        "n_fills": len(res.fills),
    }
    if res.snapshots:
        out["spread_imb"] = spread_imbalance_stats(res.snapshots)
    return out
