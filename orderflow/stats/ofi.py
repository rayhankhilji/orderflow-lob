"""Order-flow imbalance: series construction and the CKS linear regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orderflow.book.metrics import ofi
from orderflow.book.types import L2Snapshot


def compute_ofi_series(snapshots: list[L2Snapshot]) -> np.ndarray:
    """OFI between consecutive snapshots; element i is ofi(snap[i-1], snap[i])."""
    out = np.zeros(len(snapshots))
    for i in range(1, len(snapshots)):
        out[i] = ofi(snapshots[i - 1], snapshots[i])
    return out


@dataclass
class OFIFit:
    beta: float
    intercept: float
    r2: float
    t_stat: float
    n_obs: int


def ofi_regression(
    ofi_series: np.ndarray, mid_changes: np.ndarray, window: int | None = None
) -> OFIFit:
    """OLS of mid_changes on OFI (Cont–Kukanov–Stoikov impact of OFI).

    ``window``: if given, aggregate into consecutive windows of that many
    observations (sum OFI, last-first mid change) before regressing.
    """
    x = np.asarray(ofi_series, dtype=float)
    y = np.asarray(mid_changes, dtype=float)
    if window and window > 1:
        n = (len(x) // window) * window
        x = x[:n].reshape(-1, window).sum(axis=1)
        y = y[:n].reshape(-1, window).sum(axis=1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n_obs = len(y)
    dof = max(n_obs - 2, 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
    t_stat = float(coef[1] / se) if se > 0 else np.inf
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return OFIFit(
        beta=float(coef[1]),
        intercept=float(coef[0]),
        r2=r2,
        t_stat=t_stat,
        n_obs=n_obs,
    )
