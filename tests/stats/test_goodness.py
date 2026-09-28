"""Statistical-research tests: fast Hawkes residuals, GOF, OFI, Kyle lambda."""

from __future__ import annotations

import numpy as np
import pytest

from orderflow.sim.flow.hawkes import HawkesParams, simulate_hawkes
from orderflow.stats.goodness import hawkes_gof, hawkes_residuals_by_dim
from orderflow.stats.impact import kyle_lambda
from orderflow.stats.ofi import ofi_regression
from orderflow.stats.point_process import hawkes_residuals

TRUTH = HawkesParams(
    mu=np.array([1.2, 1.2, 0.35, 0.35, 0.9, 0.9]),
    alpha=np.array(
        [
            [0.8, 0.2, 0.1, 0.05, 0.1, 0.0],
            [0.2, 0.8, 0.05, 0.1, 0.0, 0.1],
            [0.2, 0.05, 0.6, 0.1, 0.05, 0.0],
            [0.05, 0.2, 0.1, 0.6, 0.0, 0.05],
            [0.2, 0.0, 0.05, 0.0, 0.5, 0.05],
            [0.0, 0.2, 0.0, 0.05, 0.05, 0.5],
        ]
    ),
    beta=5.0,
)


def test_fast_residuals_match_reference() -> None:
    rng = np.random.default_rng(1)
    tapes = simulate_hawkes(TRUTH, T=60.0, rng=rng)
    fast = hawkes_residuals_by_dim(TRUTH, tapes, T=60.0)
    ref = hawkes_residuals(TRUTH, tapes, T=60.0)
    assert np.allclose(np.concatenate(list(fast.values())), ref, atol=1e-8)


def test_fast_residuals_boundary_collision() -> None:
    # simultaneous events across dims exercise the inclusive/exclusive
    # boundary convention that broke the first incremental version
    tapes = [
        np.array([1.0, 2.0]),
        np.array([1.0, 1.5, 2.0]),
        np.array([0.5]),
        np.array([2.0]),
        np.array([1.0]),
        np.array([3.0]),
    ]
    fast = hawkes_residuals_by_dim(TRUTH, tapes, T=3.0)
    ref = hawkes_residuals(TRUTH, tapes, T=3.0)
    assert np.allclose(np.concatenate(list(fast.values())), ref, atol=1e-10)


def test_gof_truth_residuals_exponential() -> None:
    # under the true params the random time change theorem says residuals
    # are i.i.d. Exp(1); KS should not reject on a moderately long tape
    rng = np.random.default_rng(3)
    tapes = simulate_hawkes(TRUTH, T=300.0, rng=rng)
    gof = hawkes_gof(TRUTH, tapes, T=300.0)
    assert gof["n_events"] > 500
    assert gof["pooled_ks_p"] > 0.01
    for name, row in gof["by_dim"].items():
        assert row["n"] > 0, name
        assert 0.3 < row["mean_resid"] < 3.0, name


def test_gof_detects_wrong_model() -> None:
    # a Poisson-only fit (alpha=0) on clustered Hawkes data must fail GOF
    rng = np.random.default_rng(4)
    tapes = simulate_hawkes(TRUTH, T=300.0, rng=rng)
    poisson = HawkesParams(mu=TRUTH.mu * 1.5, alpha=np.zeros((6, 6)), beta=5.0)
    gof = hawkes_gof(poisson, tapes, T=300.0)
    assert gof["pooled_ks_p"] < 0.01


def test_ofi_regression_recovers_positive_impact() -> None:
    rng = np.random.default_rng(5)
    n = 2000
    beta_true, sigma = 0.4, 0.05
    x = rng.normal(0, 30, n)
    y = beta_true * x + rng.normal(0, sigma * np.abs(x).mean() * 10, n)
    fit = ofi_regression(x, y)
    assert fit.beta == pytest.approx(beta_true, rel=0.15)
    assert fit.r2 > 0.5
    assert fit.t_stat > 3
    fit_w = ofi_regression(x, y, window=10)
    assert fit_w.n_obs == n // 10


def test_kyle_lambda_positive_on_informed_flow() -> None:
    rng = np.random.default_rng(6)
    vol = rng.normal(0, 1000, 400)
    dp = 0.002 * vol + rng.normal(0, 0.5, 400)
    lam = kyle_lambda(vol, dp)
    assert lam == pytest.approx(0.002, rel=0.2)
