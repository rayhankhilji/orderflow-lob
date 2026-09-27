"""Eval harness tests: adversary mechanics, bench plumbing, selection rule."""

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.eval.adversarial import (
    Frontrunner,
    LiquidityWithdrawer,
    MomentumIgniter,
    Spoofer,
)
from orderflow.eval.execution_bench import (
    BenchConfig,
    paired_bootstrap,
    run_bench,
)
from orderflow.eval.select import exec_scores, pred_scores
from orderflow.execution.base import AlgoParticipant
from orderflow.execution.twap import TWAP
from orderflow.execution.types import ExecutionTask
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


def _sim(extra_participants=(), seed=0, horizon=60.0):
    flow_cls, params, skw = get_regime("normal", flow="zi")
    rng = np.random.default_rng(seed)
    book = LimitOrderBook(tick_size=0.01)
    seed_book(book, rng=rng, **skw)
    sim = Simulator(book, flow_cls(params), seed=seed,
                    participants=list(extra_participants))
    return sim, book


def _big_sell_exec(horizon=60.0):
    """A TWAP whose slices are big enough to trip the footprint detector."""
    task = ExecutionTask(side=Side.SELL, total_qty=3000, horizon=horizon, step=5.0)
    return AlgoParticipant(TWAP(), task), task


def test_spoofer_posts_and_pulls():
    sp = Spoofer(footprint_qty=50, spoof_qty=300, offset_ticks=3, lifetime=1.0)
    adapter, task = _big_sell_exec()
    sim, book = _sim([adapter, sp])
    adapter.algo.reset(task, book, np.random.default_rng(0))
    sim.run(62.0)  # past the last spoof's lifetime so it must be pulled
    # spoofer placed orders and cancelled them all by the end
    assert sp._next_id > 1 << 30
    assert len(sp._live) == 0


def test_igniter_and_frontrunner_fire_market_orders():
    ig = MomentumIgniter(footprint_qty=50, ignite_qty=100, interval=1.0)
    fr = Frontrunner(footprint_qty=50, front_qty=50, unwind_after=5.0)
    adapter, task = _big_sell_exec()
    sim, _book = _sim([adapter, ig, fr])
    adapter.algo.reset(task, sim.book, np.random.default_rng(0))
    sim.run(60.0)
    names = sim.participant_fills
    ig_fills = sum(f.qty for f in names.get("adv_ignite", []))
    fr_fills = sum(f.qty for f in names.get("adv_front", []))
    assert ig_fills > 0  # it fired same-direction market orders
    assert fr_fills > 0  # it traded ahead


def test_withdrawer_cancels_depth():
    wd = LiquidityWithdrawer(footprint_qty=50, cancels_per_event=3)
    adapter, task = _big_sell_exec()
    sim, book = _sim([adapter, wd])
    adapter.algo.reset(task, sim.book, np.random.default_rng(0))
    n_cancels0 = sum(1 for e in book.events if e.type.name == "CANCEL")
    sim.run(60.0)
    n_cancels1 = sum(1 for e in book.events if e.type.name == "CANCEL")
    assert n_cancels1 > n_cancels0


def test_bench_tiny_end_to_end():
    cfg = BenchConfig(flow="zi", n_episodes=2, seed=0)
    cfg.tasks = {"normal": (60.0, 300)}
    res = run_bench(algos=["twap", "ac"], regimes=["normal"], cfg=cfg,
                    verbose=False)
    table = res["regimes"]["normal"]
    assert set(table) == {"twap", "ac"}
    for s in table.values():
        assert s["completion_rate"] == 1.0
        assert np.isfinite(s["mean_is_bps"])
    assert ("normal", "ac") in res["paired_vs_twap"]


def test_paired_bootstrap_ci():
    d = np.array([1.0, 2.0, -0.5, 0.5, 1.5, 0.0])
    m, lo, hi = paired_bootstrap(d, n_boot=200, seed=0)
    assert lo <= m <= hi


def _fake_exec_bench():
    return {
        "regimes": {
            "normal": {
                "twap": {"mean_is_bps": 10.0, "cvar95_is_bps": 20.0,
                         "completion_rate": 1.0, "mean_participation": 0.1,
                         "std_is_bps": 2.0},
                "bad": {"mean_is_bps": -5.0, "cvar95_is_bps": -2.0,
                        "completion_rate": 0.5, "mean_participation": 0.1,
                        "std_is_bps": 1.0},
            },
            "exec": {
                "twap": {"mean_is_bps": 12.0, "cvar95_is_bps": 22.0,
                         "completion_rate": 1.0, "mean_participation": 0.1,
                         "std_is_bps": 2.0},
                "bad": {"mean_is_bps": -6.0, "cvar95_is_bps": -3.0,
                        "completion_rate": 1.0, "mean_participation": 0.1,
                        "std_is_bps": 1.0},
            },
        }
    }


def test_selection_requires_completion_everywhere():
    scores = exec_scores(_fake_exec_bench())
    # "bad" is cheaper on score but fails completion in "normal" -> eliminated
    assert scores["twap"]["survives"]
    assert not scores["bad"]["survives"]
    assert scores["bad"]["score"] < scores["twap"]["score"]


def test_pred_scores_rule():
    pred = {
        "normal": {
            "markov": {f"{t}_nll": 1.0 for t in
                       ("next_type", "mid_move", "spread_change",
                        "cancel_prob", "short_vol", "queue_depletion")},
            "good": {f"{t}_nll": 0.5 for t in
                     ("next_type", "mid_move", "spread_change",
                      "cancel_prob", "short_vol", "queue_depletion")},
            "weak": {f"{t}_nll": 2.0 for t in
                     ("next_type", "mid_move", "spread_change",
                      "cancel_prob", "short_vol", "queue_depletion")},
        }
    }
    ps = pred_scores(pred)
    assert ps["good"]["survives"]
    assert not ps["weak"]["survives"]
