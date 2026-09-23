import numpy as np
import pytest

from orderflow.book.book import LimitOrderBook
from orderflow.models.features import STATE_DIM, TokenStream, tokenize
from orderflow.models.targets import make_targets
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


@pytest.fixture(scope="module")
def sim_result():
    book = LimitOrderBook()
    cls, params, skw = get_regime("normal", "zi")
    seed_book(book, rng=np.random.default_rng(0), **skw)
    sim = Simulator(book, cls(params), seed=0, record_state_every_event=True)
    return sim.run(60)


def test_tokenize_alignment_and_ranges(sim_result):
    ts = tokenize(sim_result)
    n = len(sim_result.flow_events)
    assert len(ts) == n
    assert ts.type_id.min() >= 0 and ts.type_id.max() <= 5
    assert ts.level_id.min() >= 0 and ts.level_id.max() <= 5
    assert ts.size_id.min() >= 0 and ts.size_id.max() <= 7
    assert np.all(np.isfinite(ts.state))
    assert ts.state.shape == (n, STATE_DIM)
    assert np.all(np.isfinite(ts.mid_at_event))


def test_level_id_zero_when_joining_best(sim_result):
    ts = tokenize(sim_result)
    # any limit event with price at same-side best must have level 0;
    # check that limit events include level 0 somewhere
    from orderflow.book.types import EventType

    idx = [
        i
        for i, e in enumerate(sim_result.flow_events)
        if e.type is EventType.SUBMIT and not e.is_market
    ]
    assert len(idx) > 10
    assert (ts.level_id[idx] == 0).sum() > 0


def test_size_bins_reuse(sim_result):
    ts = tokenize(sim_result, B=8)
    ts2 = tokenize(sim_result, B=8, size_bins=ts.size_bins)
    assert np.array_equal(ts.size_bins, ts2.size_bins)
    assert ts2.size_id.max() <= 7


def test_targets_tiny_stream():
    # hand-constructed stream: 30 events, mid flat then up
    n = 30
    L = 5
    states = np.zeros((n, 4 * L + 1))
    states[:, 0] = 99
    states[:, L] = 10
    states[:, 2 * L] = 101
    states[:, 3 * L] = 10
    states[:20, 2 * L] = 101
    states[20:, 0] = 98  # best bid falls at event 20
    states[20:, L] = 10
    times = np.arange(n, dtype=float)
    states[:, -1] = times
    mid = np.where(np.arange(n) < 20, 100.0, 99.5)
    stream = TokenStream(
        type_id=np.zeros(n, dtype=np.int64),
        level_id=np.zeros(n, dtype=np.int64),
        size_id=np.zeros(n, dtype=np.int64),
        log_dt=np.zeros(n),
        state=np.zeros((n, STATE_DIM), dtype=np.float32),
        mid_at_event=mid,
    )

    class R:
        event_states = states
        horizon = float(n)

    tgt = make_targets(stream, R(), h_events=5)
    assert tgt["valid"].sum() == n - 5
    # mid_move: at i=10, mid[15]-mid[10]=0 -> class 1; at i=16, mid[21]-mid[16]<0 -> 0
    assert tgt["mid_move"][10] == 1
    assert tgt["mid_move"][16] == 0
    # spread_change: spread constant 2 -> encoded 0+2=2
    assert tgt["spread_change"][0] == 2
    # queue_depletion at i=15: bid price falls at 20 (within h=5) -> 1
    assert tgt["queue_depletion"][15] == 1
    assert tgt["queue_depletion"][0] == 0
    # next_type = token at i+1
    assert tgt["next_type"][0] == 0
