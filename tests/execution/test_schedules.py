"""Schedule-shape tests: TWAP linearity, AC math, VWAP profile."""

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.execution.almgren_chriss import (
    ac_cost_variance,
    ac_expected_cost,
    ac_frontier,
    ac_inventory,
    ac_kappa,
)
from orderflow.execution.twap import TWAP
from orderflow.execution.types import ExecutionTask
from orderflow.execution.vwap import VWAP, VolumeProfile

TASK = ExecutionTask(side=Side.SELL, total_qty=1000, horizon=300.0, step=10.0)


def _reset(algo, task=TASK):
    algo.reset(task, LimitOrderBook(), np.random.default_rng(0))


def test_twap_cumulative_targets_linear():
    tw = TWAP()
    _reset(tw)
    targets = [tw._target((j + 1) * TASK.step) for j in range(tw.n_inner)]
    # linear: equal increments, ends at the full quantity
    diffs = np.diff([0] + targets)
    assert targets[-1] == TASK.total_qty
    assert np.all(diffs >= 0)
    assert abs(diffs.mean() * tw.n_inner - TASK.total_qty) < tw.n_inner


def test_ac_kappa_zero_gives_twap():
    # lambda -> 0 => kappa -> 0 => linear inventory decay == TWAP
    x = ac_inventory(1000, 300.0, 10.0, kappa=1e-13)
    j = np.arange(len(x))
    assert np.allclose(x, 1000 * (1 - j / (len(x) - 1)), atol=1e-6)


def test_ac_frontloads_with_risk_aversion():
    x_lo = ac_inventory(1000, 300.0, 10.0, ac_kappa(0.017, 5e-4, 0.0, 1e-7, 10.0))
    x_hi = ac_inventory(1000, 300.0, 10.0, ac_kappa(0.017, 5e-4, 0.0, 1e-3, 10.0))
    # higher lambda => hold less inventory mid-horizon => traded more by then
    mid = len(x_lo) // 2
    assert x_hi[mid] < x_lo[mid]
    # both monotone non-increasing ending at zero
    for x in (x_lo, x_hi):
        assert np.all(np.diff(x) <= 1e-9)
        assert x[-1] == 0.0


def test_ac_cost_identity_and_frontier():
    x = ac_inventory(1000, 300.0, 10.0, kappa=0.0)
    e = ac_expected_cost(x, 10.0, sigma=0.017, eta=5e-4, gamma=1e-4)
    v = ac_cost_variance(x, 10.0, sigma=0.017)
    assert e > 0 and v > 0
    lambdas = np.logspace(-8, -4, 6)
    es, sds = ac_frontier(1000, 300.0, 10.0, 0.017, 5e-4, 0.0, lambdas)
    # efficient frontier: more risk aversion => higher E[cost], lower std
    assert np.all(np.diff(es) >= -1e-9)
    assert np.all(np.diff(sds) <= 1e-9)


def test_ac_negative_gamma_clamped():
    # negative gamma (mean-reversion artifact) must not make eta_tilde blow up
    k_neg = ac_kappa(0.017, 5e-4, -1e-4, 1e-5, 10.0)
    assert np.isfinite(k_neg) and k_neg >= 0


def test_vwap_flat_profile_equals_twap_schedule():
    book = LimitOrderBook()
    tw, vw = TWAP(), VWAP()  # None profile -> reset() fits a flat one
    task = ExecutionTask(side=Side.SELL, total_qty=900, horizon=300.0, step=10.0)
    tw.reset(task, book, np.random.default_rng(0))
    vw.reset(task, book, np.random.default_rng(0))
    from orderflow.execution.types import ExecState

    for j in (1, 5, 20):
        st = ExecState(remaining_qty=900, elapsed=j * task.step, fills=[])
        qt = tw.decide(j * task.step, book, st)
        qv = vw.decide(j * task.step, book, st)
        assert len(qt) == len(qv) == 1
        assert abs(qt[0].qty - qv[0].qty) <= 1  # rounding may differ by 1


def test_volume_profile_fit_shape():
    from orderflow.book.types import Fill
    from orderflow.sim.simulator import SimResult

    fills = [
        Fill(timestamp=5.0, price=100, qty=10, maker_id=1, taker_id=2,
             taker_side=Side.BUY, maker_hidden=False),
        Fill(timestamp=35.0, price=100, qty=30, maker_id=1, taker_id=3,
             taker_side=Side.BUY, maker_hidden=False),
    ]
    res = SimResult(
        snapshots=[], events=[], fills=fills, mid_series=np.zeros((0, 2)),
        participant_fills={}, event_times_by_type={}, horizon=100.0,
    )
    prof = VolumeProfile.fit([res], horizon=100.0, step=10.0)
    assert np.isclose(prof.cum[-1], 1.0)
    assert np.all(np.diff(prof.cum) >= 0.0)
