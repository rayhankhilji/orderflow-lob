import pytest

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Order, OrderType, Side, TimeInForce


def lim(oid, side, price, qty, ts=1.0, **kw):
    return Order(order_id=oid, side=side, price=price, qty=qty, remaining=qty, timestamp=ts, **kw)


def mkt(oid, side, qty, ts=1.0):
    return Order(
        order_id=oid,
        side=side,
        price=None,
        qty=qty,
        remaining=qty,
        timestamp=ts,
        order_type=OrderType.MARKET,
    )


@pytest.fixture
def book():
    return LimitOrderBook()


def test_resting_limit_and_l1(book):
    book.submit(lim(1, Side.BUY, 100, 10))
    book.submit(lim(2, Side.SELL, 105, 5))
    assert book.best_bid() == 100
    assert book.best_ask() == 105
    assert book.mid() == 102.5
    assert book.spread() == 5


def test_crossing_limit_fills_at_maker_price(book):
    book.submit(lim(1, Side.SELL, 100, 10))
    fills = book.submit(lim(2, Side.BUY, 102, 4))
    assert len(fills) == 1
    assert fills[0].price == 100
    assert fills[0].qty == 4
    assert fills[0].maker_id == 1
    assert fills[0].taker_id == 2
    assert book.get_order(2) is None  # fully filled taker never rests
    assert book.get_order(1).remaining == 6


def test_crossing_limit_remainder_rests_gtc(book):
    book.submit(lim(1, Side.SELL, 100, 5))
    book.submit(lim(2, Side.BUY, 100, 8))
    assert book.best_bid() == 100
    assert book.get_order(2).remaining == 3


def test_ioc_remainder_dropped(book):
    book.submit(lim(1, Side.SELL, 100, 5))
    book.submit(lim(2, Side.BUY, 100, 8, tif=TimeInForce.IOC))
    assert book.get_order(2) is None
    assert book.best_bid() is None


def test_walks_multiple_levels_fifo(book):
    book.submit(lim(1, Side.SELL, 100, 5))
    book.submit(lim(2, Side.SELL, 100, 5))  # behind id 1
    book.submit(lim(3, Side.SELL, 101, 5))
    fills = book.submit(lim(4, Side.BUY, 101, 12))
    assert [(f.maker_id, f.price, f.qty) for f in fills] == [(1, 100, 5), (2, 100, 5), (3, 101, 2)]
    assert book.get_order(3).remaining == 3


def test_market_order_walks_and_discards(book):
    book.submit(lim(1, Side.SELL, 100, 5))
    book.submit(lim(2, Side.SELL, 101, 5))
    fills = book.submit(mkt(3, Side.BUY, 20))
    assert sum(f.qty for f in fills) == 10
    assert book.get_order(3) is None
    assert book.best_ask() is None


def test_market_on_empty_side(book):
    book.submit(lim(1, Side.BUY, 100, 5))
    fills = book.submit(mkt(2, Side.BUY, 10))
    assert fills == []
    assert book.best_bid() == 100


def test_cancel_unknown_and_filled(book):
    assert book.cancel(999) is False
    book.submit(lim(1, Side.SELL, 100, 5))
    book.submit(mkt(2, Side.BUY, 5))
    assert book.cancel(1) is False  # fully filled, gone


def test_partial_cancel(book):
    book.submit(lim(1, Side.BUY, 100, 10))
    assert book.cancel(1, qty=4) is True
    assert book.get_order(1).remaining == 6
    assert book.depth(Side.BUY) == [(100, 6)]
    assert book.cancel(1) is True
    assert book.get_order(1) is None


def test_modify_qty_decrease_keeps_priority(book):
    book.submit(lim(1, Side.BUY, 100, 10))
    book.submit(lim(2, Side.BUY, 100, 10))
    book.modify(1, new_qty=5)
    assert book.queue_position(1).index == 0
    assert book.get_order(1).remaining == 5


def test_modify_qty_increase_and_price_lose_priority(book):
    book.submit(lim(1, Side.BUY, 100, 10))
    book.submit(lim(2, Side.BUY, 100, 10))
    book.modify(1, new_qty=15)
    assert book.queue_position(1).index == 1
    book.modify(2, new_price=99)
    assert book.queue_position(2).price == 99
    assert book.depth(Side.BUY) == [(100, 15), (99, 10)]


def test_modify_may_cross(book):
    book.submit(lim(1, Side.SELL, 105, 5))
    book.submit(lim(2, Side.BUY, 100, 5))
    fills = book.modify(2, new_price=105)
    assert len(fills) == 1 and fills[0].qty == 5 and fills[0].price == 105


def test_queue_position_after_fills_and_cancels(book):
    book.submit(lim(1, Side.BUY, 100, 10))
    book.submit(lim(2, Side.BUY, 100, 10))
    book.submit(lim(3, Side.BUY, 100, 10))
    qp = book.queue_position(3)
    assert (qp.index, qp.qty_ahead_visible) == (2, 20)
    book.cancel(1)
    assert book.queue_position(3).qty_ahead_visible == 10
    book.submit(lim(9, Side.SELL, 100, 5))  # fills 5 of order 2
    assert book.queue_position(3).qty_ahead_visible == 5


def test_iceberg_refresh_goes_to_back(book):
    book.submit(lim(1, Side.BUY, 100, 30, display_qty=10))
    book.submit(lim(2, Side.BUY, 100, 10))
    assert book.depth(Side.BUY) == [(100, 20)]  # only peaks/visible
    fills = book.submit(lim(3, Side.SELL, 100, 10))
    assert fills[0].maker_id == 1
    # refreshed peak now sits behind order 2
    assert book.queue_position(2).index == 0
    assert book.queue_position(1).index == 1
    assert book.depth(Side.BUY) == [(100, 20)]  # 10 (order2) + 10 (new peak)
    assert book.depth(Side.BUY, include_hidden=True) == [(100, 30)]


def test_hidden_behind_visible_priority(book):
    book.submit(lim(1, Side.SELL, 100, 5, hidden=True))
    book.submit(lim(2, Side.SELL, 100, 5))
    fills = book.submit(lim(3, Side.BUY, 100, 5))
    assert (
        fills[0].maker_id == 2
    )  # visible first despite later arrival of hidden? (hidden arrived first)
    fills = book.submit(lim(4, Side.BUY, 100, 5))
    assert fills[0].maker_id == 1
    assert fills[0].maker_hidden is True


def test_hidden_inside_spread_fills_crossing(book):
    book.submit(lim(1, Side.BUY, 100, 5))
    book.submit(lim(2, Side.SELL, 103, 5, hidden=True))  # inside a 100-105 spread
    book.submit(lim(3, Side.SELL, 105, 5))
    assert book.best_ask() == 105  # hidden cannot set L1
    assert book.depth(Side.SELL) == [(105, 5)]
    assert book.depth(Side.SELL, include_hidden=True) == [(103, 5), (105, 5)]
    fills = book.submit(lim(4, Side.BUY, 104, 5))
    assert len(fills) == 1 and fills[0].price == 103 and fills[0].maker_hidden


def test_events_tape_and_now(book):
    book.submit(lim(1, Side.SELL, 100, 5, ts=2.0))
    book.submit(mkt(2, Side.BUY, 5, ts=3.0))
    book.cancel(1)  # already filled -> no event
    types = [e.type.name for e in book.events]
    assert types == ["SUBMIT", "SUBMIT", "FILL"]
    assert book.now == 3.0
    assert len(book.fills) == 1


def test_snapshot_padding(book):
    book.submit(lim(1, Side.BUY, 100, 5))
    snap = book.snapshot(levels=3)
    assert list(snap.bid_prices) == [100, 0, 0]
    assert list(snap.bid_qtys) == [5, 0, 0]
    assert list(snap.ask_prices) == [0, 0, 0]
    assert snap.timestamp == book.now


def test_microprice(book):
    book.submit(lim(1, Side.BUY, 100, 30))
    book.submit(lim(2, Side.SELL, 110, 10))
    assert book.microprice() == pytest.approx((110 * 30 + 100 * 10) / 40)
