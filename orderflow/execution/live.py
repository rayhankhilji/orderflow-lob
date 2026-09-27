"""Live state features for execution policies and the RL environment.

``models/features.py`` builds the same 9-dim state vector offline from
recorded event states, one row per event. Here we can only afford one book
snapshot per decision step, so the rolling windows (OFI, realised vol) run
over decision-time snapshots rather than events — a coarser but honest
approximation, documented where it differs.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.book.metrics import depth_imbalance, l1_imbalance, ofi
from orderflow.book.types import L2Snapshot
from orderflow.models.features import STATE_DIM


class LiveFeatures:
    """Maintains rolling book state; produces the STATE_DIM feature vector."""

    def __init__(
        self,
        levels: int = 5,
        ofi_window: int = 20,
        rv_window: int = 50,
        tick_size: float = 0.01,
    ) -> None:
        self.levels = levels
        self.ofi_window = ofi_window
        self.rv_window = rv_window
        self.tick_size = tick_size
        self._snaps: deque[L2Snapshot] = deque(maxlen=ofi_window + 1)
        self._mids: deque[float] = deque(maxlen=rv_window + 1)
        self._depth_sum = 0.0
        self._depth_n = 0

    def update(self, book: LimitOrderBook, t: float = 0.0) -> None:
        snap = book.snapshot(self.levels)
        self._snaps.append(snap)
        mid = book.mid()
        if mid is not None and (not self._mids or mid != self._mids[-1]):
            self._mids.append(float(mid))
        bq, aq = float(snap.bid_qtys[0]), float(snap.ask_qtys[0])
        if bq + aq > 0:
            self._depth_sum += bq + aq
            self._depth_n += 1

    def state(self, time_frac: float) -> np.ndarray:
        """9-dim feature vector matching models/features.py ordering."""
        snap = self._snaps[-1] if self._snaps else L2Snapshot(
            0.0,
            np.zeros(self.levels, dtype=np.int64),
            np.zeros(self.levels, dtype=np.int64),
            np.zeros(self.levels, dtype=np.int64),
            np.zeros(self.levels, dtype=np.int64),
        )
        pb, bq = float(snap.bid_prices[0]), float(snap.bid_qtys[0])
        pa, aq = float(snap.ask_prices[0]), float(snap.ask_qtys[0])
        spread = pa - pb if pb > 0 and pa > 0 else 0.0
        l1i = l1_imbalance(snap)
        di_k = depth_imbalance(snap, self.levels)
        # rolling OFI across the snapshots we have (decision-time grid)
        ofi_w = 0.0
        snaps = list(self._snaps)
        for a, b in zip(snaps[-self.ofi_window - 1 : -1], snaps[-self.ofi_window :]):
            ofi_w += ofi(a, b)
        mid = (pb + pa) / 2 if pb > 0 and pa > 0 else 0.0
        micro_dev = 0.0
        tot = bq + aq
        if pb > 0 and pa > 0 and tot > 0:
            micro = (pa * bq + pb * aq) / tot
            micro_dev = (micro - mid) / self.tick_size
        mids = np.asarray(self._mids)
        rv = float(np.std(np.diff(mids))) if len(mids) > 2 else 0.0
        mean_depth = self._depth_sum / self._depth_n if self._depth_n else 1.0
        out = np.array(
            [
                spread,
                l1i,
                di_k,
                ofi_w,
                micro_dev,
                rv,
                time_frac,
                bq / mean_depth if mean_depth else 0.0,
                aq / mean_depth if mean_depth else 0.0,
            ],
            dtype=np.float32,
        )
        np.nan_to_num(out, copy=False)
        assert out.shape == (STATE_DIM,)
        return out

    # ---- RL observation: the 8-dim vector specced in ARCHITECTURE.md ----
    def exec_obs(
        self,
        remaining_frac: float,
        time_frac: float,
        recent_return_ticks: float = 0.0,
    ) -> np.ndarray:
        """[remaining_frac, time_frac, spread, l1_imb, depth_imb, ofi, ret, vol].

        spread in ticks/5, ofi normalised by mean top-of-book depth, ret in
        ticks/10, vol = std of decision-step mid diffs. Normalisations keep the
        policy inputs O(1) without erasing sign.
        """
        s = self.state(time_frac)
        mean_depth = (
            self._depth_sum / self._depth_n if self._depth_n else 1.0
        )
        obs = np.array(
            [
                remaining_frac,
                time_frac,
                s[0] / 5.0,  # spread
                s[1],  # l1 imbalance
                s[2],  # depth imbalance (5 levels)
                s[3] / max(mean_depth, 1.0),  # ofi
                recent_return_ticks / 10.0,
                s[5],  # rv (ticks)
            ],
            dtype=np.float32,
        )
        np.nan_to_num(obs, copy=False)
        return obs
