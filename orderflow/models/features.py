"""Order-flow tokenisation: SimResult -> aligned event tokens + state features.

Requires the simulator to have run with ``record_state_every_event=True`` so
``result.event_states`` holds the pre-event L2 rows
``[bid_prices(L), bid_qtys(L), ask_prices(L), ask_qtys(L), t]`` aligned 1:1
with ``result.flow_events``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from orderflow.book.metrics import ofi
from orderflow.book.types import EventType, L2Snapshot, Side
from orderflow.sim.simulator import EVENT_TYPES, SimResult

EVENT_CLASSES = list(EVENT_TYPES)
STATE_DIM = 9


@dataclass
class TokenStream:
    type_id: np.ndarray  # (n,) int64 in 0..5
    level_id: np.ndarray  # (n,) int64 in 0..K
    size_id: np.ndarray  # (n,) int64 in 0..B-1
    log_dt: np.ndarray  # (n,) float64
    state: np.ndarray  # (n, STATE_DIM) float32
    mid_at_event: np.ndarray  # (n,) float64, pre-event mid in ticks
    size_bins: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __len__(self) -> int:
        return len(self.type_id)


def _row_to_snap(row: np.ndarray, L: int) -> L2Snapshot:
    return L2Snapshot(
        timestamp=row[-1],
        bid_prices=row[0:L].astype(np.int64),
        bid_qtys=row[L : 2 * L].astype(np.int64),
        ask_prices=row[2 * L : 3 * L].astype(np.int64),
        ask_qtys=row[3 * L : 4 * L].astype(np.int64),
    )


def _type_id(ev) -> int:
    is_buy = ev.side is Side.BUY
    if ev.type is EventType.SUBMIT:
        if ev.is_market:
            return EVENT_CLASSES.index("buy_market" if is_buy else "sell_market")
        return EVENT_CLASSES.index("buy_limit" if is_buy else "sell_limit")
    return EVENT_CLASSES.index("buy_cancel" if is_buy else "sell_cancel")


def tokenize(
    result: SimResult,
    K: int = 5,
    size_bins: np.ndarray | None = None,
    B: int = 8,
    tick_size: float = 1.0,
) -> TokenStream:
    """Tokenise flow events + pre-event states of a recorded SimResult."""
    events = result.flow_events
    states = np.asarray(result.event_states, dtype=float)
    n = len(events)
    L = (states.shape[1] - 1) // 4 if states.ndim == 2 and states.shape[1] else 0
    if L == 0 or states.shape[0] != n:
        raise ValueError(
            "SimResult lacks aligned event_states; run with record_state_every_event=True"
        )

    type_id = np.empty(n, dtype=np.int64)
    level_id = np.zeros(n, dtype=np.int64)
    qtys = np.empty(n)
    log_dt = np.empty(n)
    mid_at = np.full(n, np.nan)
    spread = np.zeros(n)
    l1i = np.zeros(n)
    di_k = np.zeros(n)
    ofi_w = np.zeros(n)
    micro_dev = np.zeros(n)
    rv = np.zeros(n)
    bbq, abq = np.zeros(n), np.zeros(n)
    times = np.empty(n)

    prev_t = 0.0
    for i, ev in enumerate(events):
        type_id[i] = _type_id(ev)
        row = states[i]
        bp, bq = row[0], row[L]
        ap, aq = row[2 * L], row[3 * L]
        t = row[-1]
        times[i] = t
        log_dt[i] = np.log(max(t - prev_t, 0.0) + 1e-3)
        prev_t = t
        qtys[i] = ev.qty
        if bp > 0 and ap > 0:
            mid_at[i] = (bp + ap) / 2
            spread[i] = ap - bp
            tot = bq + aq
            l1i[i] = (bq - aq) / tot if tot else 0.0
            micro = (ap * bq + bp * aq) / tot if tot else mid_at[i]
            micro_dev[i] = (micro - mid_at[i]) / tick_size
        # level distance from same-side best (market orders -> 0)
        if ev.type is EventType.SUBMIT and ev.is_market:
            level_id[i] = 0
        elif ev.price is not None and ev.side is Side.BUY and bp > 0:
            level_id[i] = min(max(int(bp - ev.price), 0), K)
        elif ev.price is not None and ev.side is Side.SELL and ap > 0:
            level_id[i] = min(max(int(ev.price - ap), 0), K)
        bbq[i], abq[i] = bq, aq

    # depth imbalance over the K shallowest levels
    kmax = min(K, L)
    bsum = states[:, L : L + kmax].sum(axis=1)
    asum = states[:, 2 * L : 2 * L + kmax].sum(axis=1)
    tot = bsum + asum
    di_k = np.where(tot > 0, (bsum - asum) / np.maximum(tot, 1), 0.0)

    # rolling OFI over last 20 events
    ofis = np.zeros(n)
    for i in range(1, n):
        ofis[i] = ofi(_row_to_snap(states[i - 1], L), _row_to_snap(states[i], L))
    for i in range(n):
        lo = max(0, i - 19)
        ofi_w[i] = ofis[lo : i + 1].sum()

    # realised vol: std of last 50 mid changes (ticks)
    dmid = np.diff(mid_at, prepend=mid_at[0])
    for i in range(n):
        lo = max(0, i - 49)
        w = dmid[lo : i + 1]
        rv[i] = float(np.std(w)) if len(w) else 0.0

    time_frac = times / result.horizon if result.horizon else np.zeros(n)
    mean_depth = float(np.mean(bbq + abq)) or 1.0

    state = np.column_stack(
        [
            spread,
            l1i,
            di_k,
            ofi_w,
            micro_dev,
            rv,
            time_frac,
            bbq / mean_depth,
            abq / mean_depth,
        ]
    ).astype(np.float32)
    assert state.shape[1] == STATE_DIM

    # size bins: quantiles fit on this stream unless provided
    if size_bins is None:
        qs = np.linspace(0, 1, B + 1)[1:-1]
        size_bins = np.quantile(qtys, qs) if n else np.empty(0)
        size_bins = np.unique(size_bins)
    size_id = np.clip(np.searchsorted(size_bins, qtys), 0, B - 1).astype(np.int64)

    return TokenStream(
        type_id=type_id,
        level_id=level_id,
        size_id=size_id,
        log_dt=log_dt,
        state=state,
        mid_at_event=mid_at,
        size_bins=np.asarray(size_bins, dtype=float),
    )
