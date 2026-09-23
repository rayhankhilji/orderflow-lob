"""Label generators for the next-event model.

All targets are aligned to event index i and use a horizon of ``h_events``
events ahead. Encodings:

- ``mid_move``:      sign(mid[i+h] - mid[i]) -> {-1:0, 0:1, +1:2}
- ``spread_change``: clip(spread[i+h] - spread[i], -2, 2) + 2 -> 0..4
                     (spread from the pre-event state, in ticks)
- ``next_type`` / ``next_level`` / ``next_size`` / ``next_log_dt``:
                     token fields at i+1
- ``cancel_prob``:   1 if, scanning events i+1..i+h, a *buy cancel at level 0*
                     occurs before any *sell market* event, else 0 (proxy for
                     "the best-bid queue shrinks by cancel before a sell hits it")
- ``short_vol``:     log(RV + 1e-6), RV = sum of squared mid diffs over
                     events i+1..i+h (ticks^2)
- ``queue_depletion``: 1 if within the next h events the best-bid qty in the
                     pre-event state reaches 0 or the best-bid price falls,
                     else 0

Positions i with i + h >= n have no forward-looking label: ``valid`` masks them
(out along with the last position for ``next_*`` targets).
"""

from __future__ import annotations

import numpy as np

from orderflow.models.features import EVENT_CLASSES, TokenStream
from orderflow.sim.simulator import SimResult

BUY_CANCEL = EVENT_CLASSES.index("buy_cancel")
SELL_MARKET = EVENT_CLASSES.index("sell_market")


def make_targets(
    stream: TokenStream, result: SimResult, h_events: int = 20
) -> dict[str, np.ndarray]:
    n = len(stream)
    h = h_events
    states = np.asarray(result.event_states, dtype=float)
    L = (states.shape[1] - 1) // 4
    bid_p0 = states[:, 0]
    bid_q0 = states[:, L]
    ask_p0 = states[:, 2 * L]
    spread = np.where((bid_p0 > 0) & (ask_p0 > 0), ask_p0 - bid_p0, np.nan)
    mid = stream.mid_at_event

    valid = np.zeros(n, dtype=bool)
    valid[: max(n - h, 0)] = True  # i + h < n

    mid_move = np.ones(n, dtype=np.int64)
    spread_change = np.full(n, 2, dtype=np.int64)
    short_vol = np.zeros(n)
    queue_depl = np.zeros(n, dtype=np.int64)
    cancel_prob = np.zeros(n, dtype=np.int64)

    for i in range(n):
        if i + h >= n:
            continue
        dm = mid[i + h] - mid[i]
        mid_move[i] = 0 if dm < 0 else (2 if dm > 0 else 1)
        if np.isfinite(spread[i]) and np.isfinite(spread[i + h]):
            spread_change[i] = int(np.clip(spread[i + h] - spread[i], -2, 2)) + 2
        seg = mid[i : i + h + 1]
        rv = float(np.sum(np.diff(seg) ** 2))
        short_vol[i] = np.log(rv + 1e-6)
        # queue depletion: bid best price falls or qty hits 0 within h events
        fut_q = bid_q0[i + 1 : i + h + 1]
        fut_p = bid_p0[i + 1 : i + h + 1]
        if np.any(fut_q <= 0) or np.any(fut_p < bid_p0[i]):
            queue_depl[i] = 1
        # cancel_prob proxy: first buy_cancel@level0 vs first sell_market
        for j in range(i + 1, min(i + h + 1, n)):
            tj = stream.type_id[j]
            if tj == SELL_MARKET:
                break
            if tj == BUY_CANCEL and stream.level_id[j] == 0:
                cancel_prob[i] = 1
                break

    next_type = np.zeros(n, dtype=np.int64)
    next_level = np.zeros(n, dtype=np.int64)
    next_size = np.zeros(n, dtype=np.int64)
    next_log_dt = np.zeros(n)
    next_type[:-1] = stream.type_id[1:]
    next_level[:-1] = stream.level_id[1:]
    next_size[:-1] = stream.size_id[1:]
    next_log_dt[:-1] = stream.log_dt[1:]

    return {
        "mid_move": mid_move,
        "spread_change": spread_change,
        "next_type": next_type,
        "next_level": next_level,
        "next_size": next_size,
        "next_log_dt": next_log_dt,
        "cancel_prob": cancel_prob,
        "short_vol": short_vol,
        "queue_depletion": queue_depl,
        "valid": valid,
    }
