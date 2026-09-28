"""Hawkes goodness-of-fit: per-dimension compensator residuals + KS tests.

Under the fitted model the integrated-intensity increments between events are
i.i.d. Exp(1) (random time change theorem). ``hawkes_gof`` returns per-dim KS
statistics so a poorly fit dim is visible instead of being averaged away in
the pooled test.

O(n_i + n_j) pointer walk per (i, j) pair. With the inclusive kernel sum
A_j(t) = sum_{t_l <= t} exp(-b (t - t_l)), the contribution of dim j to the
compensator increment over (prev, t_k] is

    (alpha_ij / beta_ij) * (A_j(prev) - A_j(t_k) + n_inside)

with n_inside = #j-events in (prev, t_k]. A is updated incrementally:
A(t_k) = exp(-b dt) A(prev) + sum_{prev < t_l <= t_k} exp(-b (t_k - t_l)),
and the events advanced in the same sweep are exactly n_inside. Matches
stats.point_process.hawkes_residuals to 1e-8 (verified in tests).
"""

from __future__ import annotations

import numpy as np

from orderflow.sim.flow.hawkes import HawkesParams
from orderflow.sim.simulator import EVENT_TYPES
from orderflow.stats.point_process import ks_test_exp1


def _kernel_terms(tj: np.ndarray, ti: np.ndarray, b: float) -> np.ndarray:
    """Per-k (A_j(prev) - A_j(t_k) + n_inside); one pass over ti and tj."""
    out = np.empty(len(ti))
    ptr = 0
    a_t = 0.0  # A_j at the current boundary
    a_prev = 0.0
    prev_boundary = 0.0
    for k, tk in enumerate(ti):
        a_t *= np.exp(-b * (tk - prev_boundary))
        n_inside = 0
        while ptr < len(tj) and tj[ptr] <= tk:
            a_t += np.exp(-b * (tk - tj[ptr]))
            ptr += 1
            n_inside += 1
        out[k] = a_prev - a_t + n_inside
        a_prev = a_t
        prev_boundary = tk
    return out


def hawkes_residuals_by_dim(
    params: HawkesParams, event_times: list[np.ndarray], T: float
) -> dict[str, np.ndarray]:
    """Integrated-intensity increments per dimension (names from EVENT_TYPES)."""
    mu, alpha, beta = params.mu, params.alpha, params.beta
    n = len(event_times)
    ts = [np.asarray(t, dtype=float) for t in event_times]
    out: dict[str, np.ndarray] = {}
    for i in range(n):
        ti = ts[i]
        res = np.empty(len(ti))
        for k, tk in enumerate(ti):
            res[k] = mu[i] * (tk - (ti[k - 1] if k else 0.0))
        for j in range(n):
            res += (alpha[i, j] / beta[i, j]) * _kernel_terms(ts[j], ti, beta[i, j])
        out[EVENT_TYPES[i] if i < len(EVENT_TYPES) else f"dim{i}"] = res
    return out


def hawkes_gof(
    params: HawkesParams, event_times: list[np.ndarray], T: float
) -> dict:
    """Goodness-of-fit summary: per-dim KS + pooled KS + residual moments."""
    by_dim = hawkes_residuals_by_dim(params, event_times, T)
    table: dict[str, dict] = {}
    pooled: list[np.ndarray] = []
    for name, res in by_dim.items():
        stat, p = ks_test_exp1(res) if len(res) else (np.nan, np.nan)
        table[name] = {
            "n": len(res),
            "mean_resid": float(np.mean(res)) if len(res) else np.nan,
            "ks_stat": float(stat),
            "ks_p": float(p),
        }
        pooled.append(res)
    pool = np.concatenate(pooled) if pooled else np.empty(0)
    stat, p = ks_test_exp1(pool) if len(pool) else (np.nan, np.nan)
    return {
        "by_dim": table,
        "pooled_ks_stat": float(stat),
        "pooled_ks_p": float(p),
        "n_events": len(pool),
    }
