import random as _r

import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings

from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Order, OrderType, Side, TimeInForce


def check_invariants(book):
    bb, ba = book.best_bid(), book.best_ask()
    if bb is not None and ba is not None:
        assert bb < ba, "crossed visible book"
    for levels in (book.bids, book.asks):
        for level in levels.values():
            assert level.total_qty == sum(o.remaining for o in level.visible) + sum(
                o.remaining for o in level.hidden
            )
            assert level.total_qty > 0


def draw_op(rng, book, oid_ctr, ts_ctr):
    oid_ctr[0] += 1
    ts_ctr[0] += rng.uniform(0, 1)
    t = ts_ctr[0]
    choice = rng.randrange(10)
    if choice <= 4:  # limit submit (sometimes IOC / hidden / iceberg)
        qty = rng.randint(1, 20)
        kw = {}
        r = rng.random()
        if r < 0.15:
            kw["hidden"] = True
        elif r < 0.35:
            kw["display_qty"] = rng.randint(1, qty)
        tif = TimeInForce.IOC if rng.random() < 0.1 else TimeInForce.GTC
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        return Order(oid_ctr[0], side, 100 + rng.randint(-8, 8), qty, qty, t, tif=tif, **kw)
    if choice <= 6:  # market order
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        qty = rng.randint(1, 30)
        return Order(oid_ctr[0], side, None, qty, qty, t, order_type=OrderType.MARKET)
    if choice <= 8 and book.orders:  # cancel
        return ("cancel", rng.choice(list(book.orders)))
    if book.orders:  # modify
        return ("modify", rng.choice(list(book.orders)), rng.randint(92, 108), rng.randint(1, 25))
    return None


def apply_op(book, op):
    if op is None:
        return
    if isinstance(op, Order):
        fills = book.submit(op)
        # quantity conservation: taker filled == sum of maker fill qtys
        assert sum(f.qty for f in fills) == op.qty - op.remaining
        # fills on a walk are monotone in price (buy: non-decreasing, sell: non-increasing)
        prices = [f.price for f in fills]
        if op.side is Side.BUY:
            assert prices == sorted(prices)
        else:
            assert prices == sorted(prices, reverse=True)
        for f in fills:
            assert f.taker_id == op.order_id
    elif op[0] == "cancel":
        book.cancel(op[1])
    else:
        book.modify(op[1], new_price=op[2], new_qty=op[3])


@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.large_base_example, HealthCheck.too_slow],
)
@given(st.randoms())
def test_random_ops_invariants(random):
    assert isinstance(random, _r.Random)
    book = LimitOrderBook()
    oid_ctr, ts_ctr = [0], [0.0]
    for _ in range(200):
        apply_op(book, draw_op(random, book, oid_ctr, ts_ctr))
        check_invariants(book)
