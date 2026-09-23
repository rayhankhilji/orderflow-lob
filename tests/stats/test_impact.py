import numpy as np
import pytest

from orderflow.sim.regimes import get_regime
from orderflow.stats.impact import (
    PropagatorModel,
    calibrate_almgren_chriss,
    kyle_lambda,
    square_root_law,
)


def test_kyle_lambda():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(300) * 10
    dp = 0.05 * v + 0.01 * rng.standard_normal(300)
    lam = kyle_lambda(v, dp)
    assert lam == pytest.approx(0.05, abs=0.01)


def test_square_root_law_recovers_Y():
    rng = np.random.default_rng(0)
    Y_true, adv, sigma = 0.8, 1e5, 0.02
    Q = rng.uniform(100, 5000, 200)
    I = Y_true * sigma * np.sqrt(Q / adv) + 0.0001 * rng.standard_normal(200)
    Y = square_root_law(Q, I, adv, sigma)
    assert Y == pytest.approx(Y_true, abs=0.05)


def test_propagator_fit_runs():
    rng = np.random.default_rng(0)
    n = 5000
    signs = np.sign(rng.standard_normal(n))
    # persistent signs
    for i in range(1, n):
        if rng.random() < 0.7:
            signs[i] = signs[i - 1]
    pc = 0.02 * np.roll(signs, 1) + 0.1 * rng.standard_normal(n)
    pc[0] = 0
    m = PropagatorModel.fit(signs, pc, max_lag=30)
    assert m.G.shape == (31,)
    assert np.all(np.isfinite(m.G[1:]))


@pytest.mark.slow
def test_calibrate_almgren_chriss_runs():
    cls, params, skw = get_regime("normal", "zi")
    out = calibrate_almgren_chriss(
        flow_factory=lambda: cls(params),
        seed_kwargs=skw,
        n_runs=2,
        horizon=120.0,
        seed=0,
    )
    assert np.isfinite(out.sigma) and out.sigma > 0
    assert np.isfinite(out.eta)
    assert np.isfinite(out.gamma)
