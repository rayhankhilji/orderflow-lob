"""Implementation-shortfall accounting shared by every execution algo.

Sign convention (standard Perold IS): positive = cost vs the arrival price.
For a BUY parent:  IS = sum(p_f q_f) - arrival_mid * Q   (paid above arrival)
For a SELL parent: IS = arrival_mid * Q - sum(p_f q_f)   (sold below arrival)
i.e. IS_$ = side * (executed_value - arrival_value) in dollars.

Decomposition (exact identity, both legs in dollars):

    IS = spread_impact + timing
    spread_impact = sum_f side * (p_f - mid_f) * q_f * tick_size
    timing        = sum_f side * (mid_f - arrival_mid) * q_f * tick_size

The first leg measures how much worse than the contemporaneous mid the fills
were (half-spread paid as taker, plus temporary impact; negative for maker
fills that earn the spread). The second leg is the opportunity cost of the
mid drifting against the parent order between arrival and fill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orderflow.book.types import Fill, Side


@dataclass(slots=True)
class CostBreakdown:
    side: Side
    requested_qty: int
    filled_qty: int
    arrival_mid: float  # dollars
    avg_fill_price: float  # dollars, nan if no fills
    is_dollars: float
    is_bps: float
    spread_impact_dollars: float
    timing_dollars: float
    fill_rate: float  # filled / requested
    participation: float  # filled / total sim volume in window (nan if unknown)

    @property
    def complete(self) -> bool:
        return self.filled_qty >= self.requested_qty


def _mid_at(times: np.ndarray, mids: np.ndarray, t: float) -> float:
    idx = int(np.searchsorted(times, t, side="right")) - 1
    return float(mids[max(idx, 0)])


def cost_breakdown(
    side: Side,
    requested_qty: int,
    fills: list[Fill],
    mid_series: np.ndarray,
    arrival_mid_ticks: float,
    tick_size: float,
    total_sim_volume: float | None = None,
) -> CostBreakdown:
    """Decompose execution cost. ``mid_series`` rows are (t, mid_ticks)."""
    filled = int(sum(f.qty for f in fills))
    arrival = float(arrival_mid_ticks) * tick_size
    if filled == 0:
        return CostBreakdown(
            side=side,
            requested_qty=requested_qty,
            filled_qty=0,
            arrival_mid=arrival,
            avg_fill_price=float("nan"),
            is_dollars=0.0,
            is_bps=0.0,
            spread_impact_dollars=0.0,
            timing_dollars=0.0,
            fill_rate=0.0,
            participation=(
                filled / total_sim_volume if total_sim_volume else float("nan")
            ),
        )

    s = float(side)
    ms = np.asarray(mid_series, dtype=float).reshape(-1, 2)
    exec_value = sum(f.price * f.qty for f in fills) * tick_size
    is_dollars = s * (exec_value - arrival * filled)

    spread_impact = 0.0
    timing = 0.0
    times, mids = ms[:, 0], ms[:, 1]
    for f in fills:
        mid_f = _mid_at(times, mids, f.timestamp) * tick_size
        p_f = f.price * tick_size
        spread_impact += s * (p_f - mid_f) * f.qty
        timing += s * (mid_f - arrival) * f.qty

    notional = arrival * requested_qty
    return CostBreakdown(
        side=side,
        requested_qty=requested_qty,
        filled_qty=filled,
        arrival_mid=arrival,
        avg_fill_price=exec_value / filled,
        is_dollars=is_dollars,
        is_bps=is_dollars / notional * 1e4 if notional else float("nan"),
        spread_impact_dollars=spread_impact,
        timing_dollars=timing,
        fill_rate=filled / requested_qty if requested_qty else float("nan"),
        participation=(
            filled / total_sim_volume if total_sim_volume else float("nan")
        ),
    )


def cvar(values: np.ndarray, alpha: float = 0.95) -> float:
    """Expected shortfall: mean of the worst (1-alpha) tail."""
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) == 0:
        return float("nan")
    k = max(1, int(np.ceil((1 - alpha) * len(x))))
    return float(np.mean(x[-k:]))


def summarize(costs: list[CostBreakdown]) -> dict[str, float]:
    """Benchmark statistics over repeated runs of one algo."""
    isb = np.array([c.is_bps for c in costs], dtype=float)
    fr = np.array([c.fill_rate for c in costs], dtype=float)
    part = np.array([c.participation for c in costs], dtype=float)
    isd = np.array([c.is_dollars for c in costs], dtype=float)
    return {
        "runs": float(len(costs)),
        "mean_is_bps": float(np.mean(isb)),
        "std_is_bps": float(np.std(isb)),
        "cvar95_is_bps": cvar(isb),
        "worst_is_bps": float(np.max(isb)) if len(isb) else float("nan"),
        "mean_is_dollars": float(np.mean(isd)),
        "completion_rate": float(np.mean(fr >= 1.0 - 1e-9)),
        "mean_fill_rate": float(np.mean(fr)),
        "mean_participation": (
            float(np.nanmean(part)) if np.isfinite(part).any() else float("nan")
        ),
        "mean_spread_impact": float(np.mean([c.spread_impact_dollars for c in costs])),
        "mean_timing": float(np.mean([c.timing_dollars for c in costs])),
        # cost per unit dispersion: a simple risk-adjusted figure of merit
        "cost_over_std": float(np.mean(isb) / (np.std(isb) + 1e-12)),
    }
