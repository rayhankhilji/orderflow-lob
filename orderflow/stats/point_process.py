"""Point-process diagnostics: compensators, residuals, KS test, QQ points."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad
from scipy.stats import expon, kstest

from orderflow.sim.flow.hawkes import HawkesParams


def compensator_residuals(times: np.ndarray, intensity_fn: Callable[[float], float]) -> np.ndarray:
    """Integrated-intensity inter-arrivals: Lambda(t_k) - Lambda(t_{k-1}).

    Under the true model these are i.i.d. Exp(1) (random time change theorem).
    """
    times = np.asarray(times, dtype=float)
    out = np.empty(len(times))
    prev = 0.0
    for k, tk in enumerate(times):
        val, _ = quad(intensity_fn, prev, tk, epsabs=1e-8)
        out[k] = val
        prev = tk
    return out


def ks_test_exp1(residuals: np.ndarray) -> tuple[float, float]:
    """KS test of residuals against Exp(1); returns (stat, pvalue)."""
    return kstest(np.asarray(residuals, dtype=float), expon.cdf)


def qq_points(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(theoretical Exp(1) quantiles, empirical quantiles) for a QQ plot."""
    r = np.sort(np.asarray(residuals, dtype=float))
    probs = (np.arange(1, len(r) + 1) - 0.5) / len(r)
    return expon.ppf(probs), r


def hawkes_residuals(params: HawkesParams, event_times: list[np.ndarray], T: float) -> np.ndarray:
    """Compensator increments for each dimension, pooled into one array."""
    mu, alpha, beta = params.mu, params.alpha, params.beta
    n = len(event_times)
    all_res: list[np.ndarray] = []
    for i in range(n):
        ti = np.asarray(event_times[i], dtype=float)
        res = np.empty(len(ti))
        prev = 0.0
        for k, tk in enumerate(ti):
            inc = mu[i] * (tk - prev)
            for j in range(n):
                tj = np.asarray(event_times[j], dtype=float)
                b = beta[i, j]
                a_over_b = alpha[i, j] / b
                # events of j strictly inside (prev, tk]: 1 - exp(-b (tk - tl))
                inside = tj[(tj > prev) & (tj <= tk)]
                if len(inside):
                    inc += a_over_b * float((1.0 - np.exp(-b * (tk - inside))).sum())
                # events of j at or before prev: e^{-b(tk-tl)} - e^{-b(prev-tl)} is
                # negative of a survival difference; the compensator increment is
                # a_over_b * (e^{-b(prev - tl)} - e^{-b(tk - tl)}) for tl <= prev
                before = tj[tj <= prev]
                if len(before):
                    inc += a_over_b * float(
                        (np.exp(-b * (prev - before)) - np.exp(-b * (tk - before))).sum()
                    )
            res[k] = inc
            prev = tk
        all_res.append(res)
    return np.concatenate(all_res)
