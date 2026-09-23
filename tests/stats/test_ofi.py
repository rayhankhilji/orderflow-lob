import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.metrics import mid_series
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book
from orderflow.stats.ofi import compute_ofi_series, ofi_regression


def test_ofi_regression_positive_on_zi():
    betas, tstats = [], []
    for seed in range(3):
        cls, params, skw = get_regime("normal", "zi")
        book = LimitOrderBook()
        seed_book(book, rng=np.random.default_rng(seed), **skw)
        sim = Simulator(book, cls(params), seed=seed)
        res = sim.run(120, snapshot_every=0.5)
        snaps = res.snapshots
        ofi_s = compute_ofi_series(snaps)
        mids = mid_series(snaps)
        mid_changes = np.diff(mids, prepend=mids[0])
        fit = ofi_regression(ofi_s, mid_changes)
        betas.append(fit.beta)
        tstats.append(fit.t_stat)
    assert np.mean(betas) > 0
    assert np.mean(tstats) > 2


def test_ofi_regression_synthetic():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = 2.5 * x + 0.1 * rng.standard_normal(500)
    fit = ofi_regression(x, y)
    assert fit.beta > 0
    assert abs(fit.beta - 2.5) < 0.2
    assert fit.n_obs == 500
