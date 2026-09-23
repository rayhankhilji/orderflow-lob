"""Volatility estimators: realised variance, GARCH(1,1), LogSV, range-based."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def realized_variance(mid: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum of squared mid differences over the last ``window`` steps."""
    mid = np.asarray(mid, dtype=float)
    r = np.diff(mid)
    rv = np.full(len(mid), np.nan)
    c = np.concatenate([[0.0], np.cumsum(r**2)])
    rv[window:] = c[window:] - c[:-window]
    return rv


@dataclass
class GARCH11:
    omega: float
    alpha: float
    beta: float
    loglik: float = np.nan
    n_obs: int = 0
    _var: np.ndarray | None = None
    _last_r2: float = 0.0

    @classmethod
    def fit(cls, returns: np.ndarray, maxiter: int = 500) -> GARCH11:
        r = np.asarray(returns, dtype=float)
        n = len(r)
        var0 = float(np.var(r)) or 1e-8

        def nll(x: np.ndarray) -> float:
            lw, la, lb = x
            w, a, b = np.exp(lw), 1.0 / (1.0 + np.exp(-la)), 1.0 / (1.0 + np.exp(-lb))
            if a + b >= 1.0:
                return 1e12
            h = np.empty(n)
            h[0] = var0
            for t in range(1, n):
                h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
            if np.any(h <= 0):
                return 1e12
            return 0.5 * float(np.sum(np.log(2 * np.pi * h) + r**2 / h))

        a0, b0 = 0.1, 0.8
        w0 = var0 * (1 - a0 - b0)
        x0 = np.array([np.log(max(w0, 1e-12)), np.log(a0 / (1 - a0)), np.log(b0 / (1 - b0))])
        res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": maxiter})
        lw, la, lb = res.x
        g = cls(
            omega=float(np.exp(lw)),
            alpha=float(1 / (1 + np.exp(-la))),
            beta=float(1 / (1 + np.exp(-lb))),
            loglik=-float(res.fun),
            n_obs=n,
        )
        # store filtered variance for forecasting
        h = np.empty(n)
        h[0] = var0
        for t in range(1, n):
            h[t] = g.omega + g.alpha * r[t - 1] ** 2 + g.beta * h[t - 1]
        g._var = h
        g._last_r2 = float(r[-1] ** 2)
        return g

    def forecast(self, h: int) -> np.ndarray:
        """h-step ahead conditional variance forecast."""
        assert self._var is not None
        out = np.empty(h)
        h_next = self.omega + self.alpha * self._last_r2 + self.beta * self._var[-1]
        ab = self.alpha + self.beta
        for k in range(h):
            out[k] = h_next
            h_next = self.omega + ab * h_next
        return out


@dataclass
class LogSV:
    """log r_t^2 = h_t + xi_t; h_t = omega + phi h_{t-1} + eta_t.

    QMLE via Kalman filter (Harvey–Ruiz–Shephard): xi_t is log chi^2_1 with
    mean -1.27 and variance pi^2/2, approximated as Gaussian.
    """

    omega: float
    phi: float
    sigma_eta: float
    loglik: float = np.nan
    n_obs: int = 0
    _h_filtered: np.ndarray | None = None

    _XI_MEAN = -1.2704
    _XI_VAR = np.pi**2 / 2

    @classmethod
    def fit(cls, returns: np.ndarray, maxiter: int = 300) -> LogSV:
        r = np.asarray(returns, dtype=float)
        eps = np.finfo(float).tiny
        y = np.log(r**2 + eps) - cls._XI_MEAN  # observation = h_t + N(0, xi_var)
        n = len(y)

        def nll(x: np.ndarray) -> float:
            lo, lp, ls = x
            omega, phi, sig = lo, 1.0 / (1.0 + np.exp(-lp)) * 2 - 1, np.exp(ls)
            if not (-0.999 < phi < 0.999):
                return 1e12
            # Kalman filter for h_t | y_1..t
            m, P = 0.0, sig**2 / max(1 - phi**2, 1e-6)
            ll = 0.0
            for yt in y:
                # predict
                m_pred = omega + phi * m
                P_pred = phi**2 * P + sig**2
                # update
                F = P_pred + cls._XI_VAR
                K = P_pred / F
                innov = yt - m_pred
                ll += -0.5 * (np.log(2 * np.pi * F) + innov**2 / F)
                m = m_pred + K * innov
                P = (1 - K) * P_pred
            return -ll

        x0 = np.array([0.0, np.log(0.95 / 0.05), np.log(0.1)])
        res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": maxiter})
        lo, lp, ls = res.x
        sv = cls(
            omega=float(lo),
            phi=float(1 / (1 + np.exp(-lp)) * 2 - 1),
            sigma_eta=float(np.exp(ls)),
            loglik=-float(res.fun),
            n_obs=n,
        )
        sv._h_filtered = sv._filter(y)
        return sv

    def _filter(self, y: np.ndarray) -> np.ndarray:
        m, P = 0.0, self.sigma_eta**2 / max(1 - self.phi**2, 1e-6)
        h = np.empty(len(y))
        for t, yt in enumerate(y):
            m_pred = self.omega + self.phi * m
            P_pred = self.phi**2 * P + self.sigma_eta**2
            F = P_pred + self._XI_VAR
            K = P_pred / F
            m = m_pred + K * (yt - m_pred)
            P = (1 - K) * P_pred
            h[t] = m
        return h

    def filtered_vol(self) -> np.ndarray:
        """exp(h_t / 2): filtered per-observation volatility."""
        assert self._h_filtered is not None
        return np.exp(self._h_filtered / 2)


def parkinson(high: np.ndarray, low: np.ndarray) -> float:
    """High/low range variance estimator (per-observation variance)."""
    high, low = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    return float(np.mean(np.log(high / low) ** 2) / (4 * np.log(2)))


def garman_klass(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    o, h, l, c = (np.asarray(x, dtype=float) for x in (open_, high, low, close))
    return float(np.mean(0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2))


__all__ = [
    "GARCH11",
    "LogSV",
    "garman_klass",
    "parkinson",
    "realized_variance",
]
