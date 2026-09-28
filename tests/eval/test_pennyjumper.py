"""PennyJumper mechanics: detects passive posts at the touch, jumps 1 tick."""

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import BookEvent, EventType, Side
from orderflow.eval.adversarial import PennyJumper
from orderflow.execution.base import AlgoParticipant
from orderflow.execution.twap import TWAPPassive
from orderflow.execution.types import ExecutionTask
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


def _book(seed=0):
    _flow_cls, _params, skw = get_regime("normal", flow="zi")
    rng = np.random.default_rng(seed)
    book = LimitOrderBook(tick_size=0.01)
    seed_book(book, rng=rng, **skw)
    return book


def _submit_ev(oid: int, side: Side, price: int, qty: int, t: float = 0.0):
    return BookEvent(
        timestamp=t,
        type=EventType.SUBMIT,
        side=side,
        price=price,
        qty=qty,
        order_id=oid,
        is_market=False,
    )


def test_jumper_steps_inside_sell_post():
    book = _book()
    pj = PennyJumper(footprint_qty=50, jump_qty=60)
    ask = book.best_ask()
    orders = pj.on_event(book, _submit_ev(999, Side.SELL, ask, 300), 0.0)
    assert len(orders) == 1
    o = orders[0]
    assert o.side is Side.SELL and o.price == ask - 1 and o.qty == 60


def test_jumper_steps_inside_buy_post():
    book = _book()
    pj = PennyJumper(footprint_qty=50, jump_qty=60)
    bid = book.best_bid()
    orders = pj.on_event(book, _submit_ev(999, Side.BUY, bid, 300), 0.0)
    assert orders[0].side is Side.BUY and orders[0].price == bid + 1


def test_jumper_ignores_small_deep_market_and_own():
    book = _book()
    pj = PennyJumper(footprint_qty=50)
    ask = book.best_ask()
    # small order: no reaction
    assert pj.on_event(book, _submit_ev(1, Side.SELL, ask, 10), 0.0) == []
    # deep order (not at the touch): no reaction
    assert pj.on_event(book, _submit_ev(2, Side.SELL, ask + 5, 300), 0.0) == []
    # market orders: no reaction
    ev = _submit_ev(3, Side.SELL, None, 500)
    ev.is_market = True
    assert pj.on_event(book, ev, 0.0) == []
    # its own ids: no reaction
    pj._my_ids.add(7)
    assert pj.on_event(book, _submit_ev(7, Side.SELL, ask, 300), 0.0) == []


def test_jumper_expires_orders_in_sim():
    pj = PennyJumper(footprint_qty=50, jump_qty=60, lifetime=1.0)
    task = ExecutionTask(side=Side.SELL, total_qty=3000, horizon=60.0, step=5.0)
    adapter = AlgoParticipant(TWAPPassive(), task)
    flow_cls, params, skw = get_regime("normal", flow="zi")
    rng = np.random.default_rng(0)
    book = LimitOrderBook(tick_size=0.01)
    seed_book(book, rng=rng, **skw)
    sim = Simulator(book, flow_cls(params), seed=0,
                    participants=[adapter, pj])
    adapter.algo.reset(task, book, np.random.default_rng(0))
    sim.run(65.0)
    assert len(pj._live) == 0
