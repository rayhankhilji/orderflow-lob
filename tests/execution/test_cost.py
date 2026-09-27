"""Cost accounting: sign conventions and the exact decomposition."""

import numpy as np

from orderflow.book.types import Fill, Side
from orderflow.execution.cost import cost_breakdown, cvar, summarize


def _fill(t, price, qty, taker_side=Side.BUY):
    return Fill(
        timestamp=t, price=price, qty=qty, maker_id=-1, taker_id=1 << 40,
        taker_side=taker_side, maker_hidden=False,
    )


def test_sell_below_arrival_is_positive_cost():
    # arrival mid 10000 ticks ($100); we sold at 9995 -> positive IS
    ms = np.array([[0.0, 10000.0], [10.0, 9995.0], [20.0, 9990.0]])
    cb = cost_breakdown(
        Side.SELL, 100, [_fill(10.0, 9995, 100)], ms,
        arrival_mid_ticks=10000.0, tick_size=0.01,
        total_sim_volume=1000.0,
    )
    assert cb.is_dollars > 0  # sold below arrival = cost
    assert cb.is_bps > 0
    assert cb.complete
    # identity: IS = spread_impact + timing
    assert np.isclose(
        cb.is_dollars, cb.spread_impact_dollars + cb.timing_dollars, atol=1e-9
    )


def test_buy_above_arrival_is_positive_cost():
    cb = cost_breakdown(
        Side.BUY, 50, [_fill(5.0, 10005, 50, taker_side=Side.SELL)],
        np.array([[0.0, 10000.0], [5.0, 10002.0]]),
        arrival_mid_ticks=10000.0, tick_size=0.01,
    )
    assert cb.is_dollars > 0
    # timing leg: mid rose 2 ticks before we bought -> adverse, positive
    assert cb.timing_dollars > 0


def test_decomposition_sums_across_fills():
    fills = [_fill(10.0, 9995, 60), _fill(20.0, 9990, 40)]
    ms = np.array([[0.0, 10000.0], [10.0, 9994.0], [20.0, 9988.0]])
    cb = cost_breakdown(
        Side.SELL, 100, fills, ms, 10000.0, 0.01, total_sim_volume=500.0
    )
    assert np.isclose(
        cb.is_dollars, cb.spread_impact_dollars + cb.timing_dollars, atol=1e-9
    )
    assert np.isclose(cb.participation, 100 / 500.0)
    assert cb.avg_fill_price < cb.arrival_mid


def test_cvar_and_summarize():
    vals = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    assert cvar(vals, 0.8) >= np.mean([10.0, 20.0]) - 1e-9
    cbs = [
        cost_breakdown(
            Side.SELL, 100, [_fill(5.0, 9990, 100)],
            np.array([[0.0, 10000.0]]), 10000.0, 0.01,
        ),
        cost_breakdown(
            Side.SELL, 100, [_fill(5.0, 9980, 100)],
            np.array([[0.0, 10000.0]]), 10000.0, 0.01,
        ),
    ]
    s = summarize(cbs)
    assert s["runs"] == 2
    assert s["completion_rate"] == 1.0
    assert s["mean_is_bps"] > 0
    assert s["worst_is_bps"] >= s["mean_is_bps"]
