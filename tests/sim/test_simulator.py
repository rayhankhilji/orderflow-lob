import numpy as np
import pytest

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Order, OrderType, Side
from orderflow.sim.actions import CancelOrder
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


def make_sim(seed=0, flow="zi", regime="normal", participants=()):
    book = LimitOrderBook()
    cls, params, skw = get_regime(regime, flow=flow)
    seed_book(book, rng=np.random.default_rng(seed), **skw)
    return Simulator(book, cls(params), seed=seed, participants=participants)


def test_seed_book_not_crossed():
    book = LimitOrderBook()
    _, _, skw = get_regime("normal", "zi")
    seed_book(book, rng=np.random.default_rng(0), **skw)
    assert book.best_bid() < book.best_ask()
    assert book.spread() == 1


def test_reproducibility():
    sim1, sim2 = make_sim(seed=7), make_sim(seed=7)
    r1, r2 = sim1.run(30), sim2.run(30)
    tape1 = [(e.type, e.side, e.price, e.qty, e.timestamp) for e in r1.events]
    tape2 = [(e.type, e.side, e.price, e.qty, e.timestamp) for e in r2.events]
    assert tape1 == tape2
    assert len(tape1) > 50


def test_mid_series_monotone_in_t():
    res = make_sim(seed=1).run(60)
    t = res.mid_series[:, 0]
    assert np.all(np.diff(t) > 0)


def test_participant_fills_tagged():
    class Aggressor:
        name = "aggro"

        def on_event(self, book, event, t):
            return []

        def next_wakeup(self, t):
            return 5.0 if self.sent == 0 else None

        sent = 0

        def on_time(self, book, t):
            self.sent += 1
            return [
                Order(
                    order_id=0,
                    side=Side.BUY,
                    price=None,
                    qty=3,
                    remaining=3,
                    timestamp=t,
                    order_type=OrderType.MARKET,
                )
            ]

    p = Aggressor()
    res = make_sim(participants=[p]).run(30)
    fills = res.participant_fills["aggro"]
    assert fills and all(f.taker_side == Side.BUY for f in fills)


def test_wakeup_times_and_cancel_request():
    woke = []

    class Timer:
        name = "timer"

        def on_event(self, book, event, t):
            return []

        def next_wakeup(self, t):
            nxt = 1.0 * (len(woke) + 1)
            return nxt if nxt <= 3.0 else None

        def on_time(self, book, t):
            woke.append(t)
            return []

    make_sim(participants=[Timer()]).run(10)
    assert woke == [1.0, 2.0, 3.0]


def test_cancel_order_participant_request():
    class Canceller:
        name = "cxl"
        done = False

        def on_event(self, book, event, t):
            return []

        def next_wakeup(self, t):
            return None if self.done else 1.0

        def on_time(self, book, t):
            self.done = True
            depth = book.depth(Side.BUY, 1)
            level = book.level(Side.BUY, depth[0][0])
            return [CancelOrder(level.visible[0].order_id)]

    sim = make_sim(participants=[Canceller()])
    res = sim.run(5)
    assert any(e.type.name == "CANCEL" for e in res.events)


@pytest.mark.parametrize("flow", ["zi", "hawkes", "queue_reactive"])
def test_flows_run_normal(flow):
    sim = make_sim(flow=flow)
    res = sim.run(60)
    book = sim.book
    if book.best_bid() is not None and book.best_ask() is not None:
        assert book.best_bid() < book.best_ask()
    # every event type present
    nonempty = [k for k, v in res.event_times_by_type.items() if len(v) > 0]
    assert len(nonempty) == 6, res.event_times_by_type


def test_hawkes_branching_assertion():
    from orderflow.sim.flow.hawkes import HawkesFlow, HawkesParams

    bad = HawkesParams(mu=[0.1] * 6, alpha=np.full((6, 6), 1.0), beta=1.0)
    with pytest.raises(AssertionError):
        HawkesFlow(bad)
