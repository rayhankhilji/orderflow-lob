import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.sim.actions import SubmitMarket
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


def make_recorded(seed=0, levels=5):
    book = LimitOrderBook()
    cls, params, skw = get_regime("normal", "zi")
    seed_book(book, rng=np.random.default_rng(seed), **skw)
    return Simulator(
        book, cls(params), seed=seed, record_state_every_event=True, state_levels=levels
    )


def test_event_states_align_with_flow_events():
    res = make_recorded().run(30)
    assert len(res.flow_events) > 50
    assert res.event_states.shape == (len(res.flow_events), 4 * 5 + 1)
    assert np.all(np.diff(res.event_states[:, -1]) >= 0)


def test_state_precedes_event():
    # state must show pre-event ask qty for a market buy
    res = make_recorded().run(30)
    L = 5
    for i, ev in enumerate(res.flow_events):
        if ev.is_market and ev.side is Side.BUY:
            ask_q_before = res.event_states[i, 3 * L]
            assert ask_q_before > 0
            break
    else:
        raise AssertionError("no market buy in tape")


def test_pending_wakeup_monotone():
    # a stale wakeup must not move the clock backwards
    class Stale:
        name = "stale"

        def on_event(self, book, event, t):
            return []

        def next_wakeup(self, t):
            return None if self.hits else max(t - 1.0, 0.0)  # stale once

        def on_time(self, book, t):
            self.hits.append(t)
            return []

        def __init__(self):
            self.hits = []

    p = Stale()
    sim = make_recorded(seed=1)
    sim.participants = [p]
    sim.run(5)
    ts = [e.timestamp for e in sim.book.events]
    assert p.hits  # fired once, clock never rewound
    assert ts == sorted(ts)


class _ForcedMarket:
    """Deterministic flow for the snapshot-cadence test."""

    def __init__(self, gap):
        self.gap = gap

    def next_event(self, book, t, rng):
        return self.gap, SubmitMarket(Side.BUY, 1)


def test_snapshot_cadence_catches_up():
    book = LimitOrderBook()
    seed_book(book, 10_000, 5, 50, np.random.default_rng(0))
    sim = Simulator(book, _ForcedMarket(gap=7.0), seed=0)
    res = sim.run(50, snapshot_every=1.0)
    # events land at t=7,14,...: one snapshot per event, times must not burst
    assert len(res.snapshots) >= 6
