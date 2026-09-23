"""Multivariate exponential-kernel Hawkes MLE.

Log-likelihood uses the O(n) recursion per dimension i:
    R_ij(k) = exp(-beta_ij (t_k^i - t_{k-1}^i))
              * (R_ij(k-1) + sum_{t_l^j in [t_{k-1}^i, t_k^i)} e^{-beta_ij (t_k^i - t_l^j)})
    lambda_i(t_k^i) = mu_i + sum_j R_ij(k)
Compensator:
    Lambda_i(T) = mu_i T + sum_j (alpha_ij/beta_ij)
                  * sum_l (1 - exp(-beta_ij (T - t_l^j)))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from orderflow.sim.flow.hawkes import HawkesParams


@dataclass
class HawkesFit:
    params: HawkesParams
    loglik: float
    n_obs: int
    converged: bool


def branching_ratio(params: HawkesParams) -> float:
    return params.branching_ratio()


def simulate(params: HawkesParams, T: float, rng: np.random.Generator) -> list[np.ndarray]:
    from orderflow.sim.flow.hawkes import simulate_hawkes

    return simulate_hawkes(params, T, rng)


def _loglik(
    event_times: list[np.ndarray], T: float, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> float:
    """Total log-likelihood of the multivariate Hawkes process on [0, T]."""
    n = len(event_times)
    ll = 0.0
    for i in range(n):
        ti = event_times[i]
        R = np.zeros(n)
        ptrs = np.zeros(n, dtype=int)
        prev = 0.0
        lam_sum = 0.0
        for k in range(len(ti)):
            tk = ti[k]
            decay = np.exp(-beta[i] * (tk - prev))
            acc = np.zeros(n)
            for j in range(n):
                tj = event_times[j]
                while ptrs[j] < len(tj) and tj[ptrs[j]] < tk:
                    acc[j] += np.exp(-beta[i, j] * (tk - tj[ptrs[j]]))
                    ptrs[j] += 1
            R = decay * R + acc
            lam = mu[i] + float((alpha[i] * R).sum())
            if lam <= 0:
                return -np.inf
            lam_sum += np.log(lam)
            prev = tk
        comp = mu[i] * T
        for j in range(n):
            tj = event_times[j]
            if len(tj):
                comp += (alpha[i, j] / beta[i, j]) * float(
                    (1.0 - np.exp(-beta[i, j] * (T - tj))).sum()
                )
        ll += lam_sum - comp
    return ll


def fit_hawkes_exp(
    event_times: list[np.ndarray],
    T: float,
    beta: float | None = None,
    maxiter: int = 500,
    rng: np.random.Generator | None = None,
) -> HawkesFit:
    """MLE for the 6-dim (or n-dim) exponential Hawkes on [0, T].

    Parameters are optimised in log space (positivity guaranteed).
    ``beta`` given => single shared decay held fixed (robust default);
    ``beta=None`` => shared beta is also estimated.
    """
    rng = rng or np.random.default_rng(0)
    n = len(event_times)
    event_times = [np.asarray(e, dtype=float) for e in event_times]
    n_obs = int(sum(len(e) for e in event_times))

    # init: mu from mean rates, alpha small
    rates = np.array([max(len(e), 1) / T for e in event_times])
    mu0 = rates * 0.6
    alpha0 = np.full((n, n), 0.3 / n) + np.eye(n) * 0.1
    beta0 = beta if beta is not None else 1.0
    x0 = np.concatenate(
        [np.log(mu0), np.log(alpha0).ravel(), [] if beta is not None else [np.log(beta0)]]
    )

    def unpack(x: np.ndarray):
        mu = np.exp(x[:n])
        alpha = np.exp(x[n : n + n * n]).reshape(n, n)
        b = np.full((n, n), beta) if beta is not None else np.full((n, n), np.exp(x[-1]))
        return mu, alpha, b

    def nll(x: np.ndarray) -> float:
        mu, alpha, b = unpack(x)
        v = -_loglik(event_times, T, mu, alpha, b)
        return v if np.isfinite(v) else 1e12

    res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": maxiter})
    mu, alpha, b = unpack(res.x)
    params = HawkesParams(mu=mu, alpha=alpha, beta=b)
    return HawkesFit(
        params=params, loglik=-float(res.fun), n_obs=n_obs, converged=bool(res.success)
    )
