import numpy as np
import pytest

from orderflow.book.metrics import (
    adverse_selection,
    depth_imbalance,
    effective_spread,
    l1_imbalance,
    ofi,
    realized_spread,
    spread_series,
)
from orderflow.book.types import Fill, L2Snapshot, Side


def snap(bp, bq, ap, aq, n=2):
    pad = lambda a: np.array(list(a) + [0] * (n - len(a)))
    return L2Snapshot(
        timestamp=0.0, bid_prices=pad(bp), bid_qtys=pad(bq), ask_prices=pad(ap), ask_qtys=pad(aq)
    )


def test_l1_imbalance():
    s = snap([100], [30], [101], [10])
    assert l1_imbalance(s) == pytest.approx(0.5)


def test_l1_imbalance_empty_side():
    s = snap([100], [30], [], [])
    assert l1_imbalance(s) == 1.0
    s2 = snap([], [], [], [])
    assert l1_imbalance(s2) == 0.0


def test_depth_imbalance():
    s = snap([100, 99], [10, 20], [101], [15])
    assert depth_imbalance(s, levels=2) == pytest.approx((30 - 15) / 45)


def test_ofi_known_answer():
    # bid improves in price (100->101), qty 10->12; ask unchanged price, qty grows 8->10
    prev = snap([100], [10], [102], [8])
    nxt = snap([101], [12], [102], [10])
    # e = +qb_n (bid up) - 0 (Pb_n > Pb_{n-1}, second term needs <=) ... apply formula:
    # bid: 1{101>=100}*12 - 1{101<=100}*10 = 12
    # ask: -1{102<=102}*10 + 1{102>=102}*8 = -10 + 8 = -2
    assert ofi(prev, nxt) == 10.0


def test_ofi_bid_drops():
    prev = snap([100], [10], [102], [8])
    nxt = snap([99], [5], [102], [8])
    # bid: 0*5 - 1*10 = -10 ; ask: -0*8 + 1*8 = 8? pa_n==pa_p: -1{102<=102}*8 + 1{102>=102}*8 = 0
    assert ofi(prev, nxt) == -10.0


def test_spread_series_nan_on_empty():
    snaps = [snap([100], [5], [102], [5]), snap([], [], [102], [5])]
    out = spread_series(snaps)
    assert out[0] == 2 and np.isnan(out[1])


def test_effective_and_realized_spread():
    f = Fill(
        timestamp=0,
        price=102,
        qty=5,
        maker_id=1,
        taker_id=2,
        taker_side=Side.BUY,
        maker_hidden=False,
    )
    assert effective_spread(f, 101.0) == 2.0
    assert realized_spread(f, 100.5) == 3.0  # mid fell: maker kept more


def test_adverse_selection_sign():
    fills = [
        Fill(
            timestamp=1.0,
            price=100,
            qty=5,
            maker_id=1,
            taker_id=2,
            taker_side=Side.BUY,
            maker_hidden=False,
        ),
        Fill(
            timestamp=1.0,
            price=100,
            qty=5,
            maker_id=3,
            taker_id=4,
            taker_side=Side.SELL,
            maker_hidden=False,
        ),
    ]
    times = np.array([0.0, 1.0, 2.0, 3.0])
    mids = np.array([100.0, 100.0, 102.0, 102.0])  # mid rises after t=1
    arr, mean = adverse_selection(fills, times, mids, horizon_seconds=1.0)
    assert arr[0] == pytest.approx(2.0)  # buy taker informed
    assert arr[1] == pytest.approx(-2.0)  # sell taker adversely moved
    assert mean == pytest.approx(0.0)
