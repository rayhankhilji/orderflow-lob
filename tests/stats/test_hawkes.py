import numpy as np
import pytest

from orderflow.sim.flow.hawkes import HawkesParams, simulate_hawkes
from orderflow.stats.hawkes import fit_hawkes_exp
from orderflow.stats.point_process import hawkes_residuals, ks_test_exp1


@pytest.fixture(scope="module")
def hawkes_data():
    mu = np.array([0.5, 0.5])
    alpha = np.array([[0.3, 0.1], [0.1, 0.3]])
    beta = 2.0
    full = HawkesParams(
        mu=np.concatenate([mu, np.zeros(4)]),
        alpha=np.pad(alpha, ((0, 4), (0, 4))),
        beta=beta,
    )
    rng = np.random.default_rng(0)
    ev6 = simulate_hawkes(full, T=4000, rng=rng)
    # the fitted model is 2-dim over the first two types
    return [ev6[0], ev6[1]], mu, alpha, beta


def test_hawkes_mle_recovery(hawkes_data):
    event_times, mu, alpha, beta = hawkes_data
    fit = fit_hawkes_exp(event_times, T=4000, beta=beta, rng=np.random.default_rng(0))
    est_mu = fit.params.mu
    est_alpha = fit.params.alpha
    for i in range(2):
        assert est_mu[i] == pytest.approx(mu[i], rel=0.2)
        for j in range(2):
            assert est_alpha[i, j] == pytest.approx(alpha[i, j], rel=0.2)


def test_hawkes_residuals_exp1(hawkes_data):
    event_times, mu, alpha, beta = hawkes_data
    params = HawkesParams(mu=mu, alpha=alpha, beta=beta)
    res = hawkes_residuals(params, event_times, T=4000)
    _, p = ks_test_exp1(res)
    assert p > 0.01, f"KS p={p}"
