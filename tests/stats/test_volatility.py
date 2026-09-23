import numpy as np
import pytest

from orderflow.stats.volatility import GARCH11, LogSV, garman_klass, parkinson, realized_variance


def test_realized_variance():
    mid = np.array([100.0, 101.0, 100.0, 102.0])
    rv = realized_variance(mid, 2)
    assert np.isnan(rv[1])
    assert rv[2] == pytest.approx(1.0 + 1.0)  # r1^2 + r2^2
    assert rv[3] == pytest.approx(1.0 + 4.0)


def test_garch_recovery():
    rng = np.random.default_rng(0)
    n = 5000
    w, a, b = 0.1, 0.1, 0.8
    h = np.empty(n)
    r = np.empty(n)
    h[0] = w / (1 - a - b)
    r[0] = np.sqrt(h[0]) * rng.standard_normal()
    for t in range(1, n):
        h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_normal()
    g = GARCH11.fit(r)
    assert g.alpha + g.beta == pytest.approx(a + b, abs=0.1)
    fc = g.forecast(5)
    assert np.all(np.isfinite(fc)) and np.all(fc > 0)


def test_logsv_runs_finite():
    rng = np.random.default_rng(1)
    r = 0.01 * rng.standard_normal(800)
    sv = LogSV.fit(r)
    vols = sv.filtered_vol()
    assert vols.shape == (800,)
    assert np.all(np.isfinite(vols)) and np.all(vols > 0)


def test_range_estimators_positive():
    rng = np.random.default_rng(2)
    base = 100 + np.cumsum(0.05 * rng.standard_normal(500))
    high = base + np.abs(0.05 * rng.standard_normal(500))
    low = base - np.abs(0.05 * rng.standard_normal(500))
    close = base
    open_ = np.roll(base, 1)
    assert parkinson(high, low) > 0
    assert garman_klass(open_, high, low, close) > 0
