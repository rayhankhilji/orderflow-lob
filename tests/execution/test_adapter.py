"""End-to-end adapter tests: does an ExecutionAlgo actually complete in sim?"""

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.execution.base import AlgoParticipant
from orderflow.execution.twap import TWAP, TWAPPassive
from orderflow.execution.types import ExecutionTask
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


def _run(algo, task, regime="normal", flow="zi", seed=0, extra_time=2.0):
    flow_cls, params, seed_kwargs = get_regime(regime, flow=flow)
    rng = np.random.default_rng(seed)
    book = LimitOrderBook(tick_size=0.01)
    seed_book(book, rng=rng, **seed_kwargs)
    algo.reset(task, book, rng)
    adapter = AlgoParticipant(algo, task)
    sim = Simulator(book, flow_cls(params), seed=seed, participants=[adapter])
    res = sim.run(task.horizon + extra_time)
    return adapter, res, book


def test_twap_completes_market():
    task = ExecutionTask(side=Side.SELL, total_qty=300, horizon=60.0, step=5.0)
    adapter, res, _ = _run(TWAP(), task)
    filled = sum(f.qty for f in adapter._fills)
    assert filled == task.total_qty
    assert adapter.remaining == 0
    # attribution: participant_fills tags the algo name
    assert sum(f.qty for f in res.participant_fills["twap"]) == filled
    # exec order ids are all from the adapter's private range
    assert all(o >= (1 << 40) for o in adapter._my_ids)


def test_twap_passive_no_resting_leftovers():
    task = ExecutionTask(side=Side.SELL, total_qty=50, horizon=60.0, step=5.0)
    adapter, _res, book = _run(TWAPPassive(), task)
    # by the deadline everything is liquidated (market fallback)
    assert adapter.remaining == 0
    assert len(adapter._open) == 0
    # no exec order may still be resting: book.orders only holds live orders
    assert all(oid not in book.orders for oid in adapter._my_ids)


def test_deadline_liquidation_when_flow_stops():
    # horizon longer than sim clock allows few decisions; the deadline step
    # must still close out the remainder
    task = ExecutionTask(side=Side.SELL, total_qty=120, horizon=20.0, step=5.0)
    adapter, _res, _ = _run(TWAP(), task, extra_time=1.0)
    assert adapter.remaining == 0


def test_arrival_mid_recorded():
    task = ExecutionTask(side=Side.SELL, total_qty=30, horizon=30.0, step=5.0)
    adapter, _, _ = _run(TWAP(), task)
    assert adapter._arrival_mid is not None and adapter._arrival_mid > 0
