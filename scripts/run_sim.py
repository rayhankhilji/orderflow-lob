"""Run a simulation and print microstructure summary stats.

Usage: python scripts/run_sim.py --flow hawkes --regime normal --seed 0 --horizon 300
"""

from __future__ import annotations

import argparse

import numpy as np

from orderflow.book.book import LimitOrderBook
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="hawkes", choices=["hawkes", "zi", "queue_reactive"])
    ap.add_argument("--regime", default="normal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--tick-size", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    flow_cls, params, seed_kwargs = get_regime(args.regime, flow=args.flow)
    rng = np.random.default_rng(args.seed)
    book = LimitOrderBook(tick_size=args.tick_size)
    seed_book(book, rng=rng, **seed_kwargs)
    flow = flow_cls(params)
    sim = Simulator(book, flow, seed=args.seed)
    res = sim.run(args.horizon, snapshot_every=1.0)

    n_events = len(res.events)
    eps = n_events / args.horizon
    spreads = [
        s.ask_prices[0] - s.bid_prices[0]
        for s in res.snapshots
        if s.bid_prices[0] and s.ask_prices[0]
    ]
    ms = res.mid_series
    # realised vol: std of per-minute mid diffs, in bps/min
    vol_bps = np.nan
    if len(ms) > 2:
        grid = np.arange(0, ms[-1, 0], 60.0)
        idx = np.clip(np.searchsorted(ms[:, 0], grid, side="right") - 1, 0, len(ms) - 1)
        mins = ms[idx, 1]
        if len(mins) > 2:
            ret = np.diff(mins) / mins[:-1] * 1e4
            vol_bps = float(np.std(ret))
    depth10 = (
        float(np.mean([s.bid_qtys.sum() + s.ask_qtys.sum() for s in res.snapshots]))
        if res.snapshots
        else np.nan
    )
    print(f"flow={args.flow} regime={args.regime} seed={args.seed} horizon={args.horizon}s")
    print(f"  events/sec        {eps:.2f}  (total {n_events})")
    print(f"  mean spread       {np.mean(spreads):.2f} ticks" if spreads else "  mean spread  n/a")
    print(f"  realised vol      {vol_bps:.2f} bps/min")
    print(f"  mean depth (10L)  {depth10:.0f} shares")
    print(f"  fills             {len(res.fills)}")
    for k, v in res.event_times_by_type.items():
        print(f"    {k:12s} {len(v)}")


if __name__ == "__main__":
    main()
