"""Stylised-fact measurements sanity checks."""

import numpy as np

from orderflow.book.types import L2Snapshot
from orderflow.stats.stylized import (
    acf,
    resample_mid,
    return_moments,
    spread_imbalance_stats,
)


def test_return_moments_gaussian():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(20000)
    m = return_moments(r)
    assert abs(m["excess_kurtosis"]) < 0.15
    assert abs(m["skew"]) < 0.1


def test_acf_known_series():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    # an AR(1) with phi=0.7 has rho(1)~0.7, rho(2)~0.49
    y = np.zeros(5000)
    for i in range(1, 5000):
        y[i] = 0.7 * y[i - 1] + x[i]
    a = acf(y, 2)
    assert 0.6 < a[0] < 0.8
    assert 0.35 < a[1] < 0.6


def test_resample_mid_last_observation():
    ms = np.array([[0.0, 100.0], [2.5, 101.0], [4.0, 99.0]])
    grid = resample_mid(ms, dt=1.0)
    # grid t=0..3 uses last observation <= t (t=4.0 obs is past the grid)
    assert grid[0] == 100.0 and grid[2] == 100.0 and grid[3] == 101.0


def test_spread_imbalance_stats_finite():
    snaps = [
        L2Snapshot(
            float(i),
            np.array([100, 99], dtype=np.int64),
            np.array([40 + i, 30], dtype=np.int64),
            np.array([101 + (i % 3), 102], dtype=np.int64),  # varying spread
            np.array([30, 30], dtype=np.int64),
        )
        for i in range(10)
    ]
    s = spread_imbalance_stats(snaps)
    assert np.isfinite(s["mean_spread"])
    assert np.isfinite(s["corr_spread_absimb"])
