"""Run an execution algo on a simulated regime and print the cost breakdown.

Usage:
    python scripts/run_execution.py --algo twap --regime exec --qty 100000 \
        --horizon 1800 --step 10 --side sell --runs 3
    python scripts/run_execution.py --algo ac --ac-params artifacts/params/ac.json
    python scripts/run_execution.py --algo rl --policy artifacts/models/ppo_exec.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import orderflow.execution  # noqa: F401 - registers algos
from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.execution.base import AlgoParticipant
from orderflow.execution.cost import cost_breakdown, summarize
from orderflow.execution.types import ExecutionTask
from orderflow.execution.vwap import VolumeProfile
from orderflow.registry import get_execution
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


def build_algo(name: str, args):
    cls = get_execution(name)
    if name == "ac":
        algo = cls(lam=args.lam)
        if args.ac_params:
            with open(args.ac_params) as fh:
                p = json.load(fh)
            algo.sigma, algo.eta, algo.gamma = p["sigma"], p["eta"], p["gamma"]
        return algo
    if name == "vwap":
        algo = cls()
        if args.vwap_profile:
            algo.profile = VolumeProfile(np.load(args.vwap_profile))
        return algo
    if name == "rl":
        algo = cls()
        if args.policy:
            algo.load_policy(args.policy)
        return algo
    return cls()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="twap")
    ap.add_argument("--flow", default="hawkes")
    ap.add_argument("--regime", default="exec")
    ap.add_argument("--side", default="sell", choices=["buy", "sell"])
    ap.add_argument("--qty", type=int, default=100_000)
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--tick-size", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--ac-params", default=None)
    ap.add_argument("--vwap-profile", default=None)
    ap.add_argument("--policy", default=None)
    args = ap.parse_args()

    side = Side.SELL if args.side == "sell" else Side.BUY
    flow_cls, params, seed_kwargs = get_regime(args.regime, flow=args.flow)
    costs = []
    for run in range(args.runs):
        seed = args.seed + run
        rng = np.random.default_rng(seed)
        book = LimitOrderBook(tick_size=args.tick_size)
        seed_book(book, rng=rng, **seed_kwargs)
        algo = build_algo(args.algo, args)
        task = ExecutionTask(
            side=side,
            total_qty=args.qty,
            horizon=args.horizon,
            step=args.step,
            start_time=0.0,
        )
        adapter = AlgoParticipant(algo, task)
        algo.reset(task, book, np.random.default_rng(seed + 1))
        sim = Simulator(book, flow_cls(dict(params)), seed=seed, participants=[adapter])
        res = sim.run(args.horizon + 1.0)
        arrival = adapter._arrival_mid or book.mid() or 0.0
        cb = cost_breakdown(
            side=side,
            requested_qty=args.qty,
            fills=adapter._fills,
            mid_series=res.mid_series,
            arrival_mid_ticks=arrival,
            tick_size=args.tick_size,
            total_sim_volume=float(sum(f.qty for f in res.fills)),
        )
        costs.append(cb)
        print(
            f"run {run}: filled {cb.filled_qty}/{cb.requested_qty} "
            f"IS={cb.is_dollars:+.2f}$ ({cb.is_bps:+.2f} bps) "
            f"spread+impact={cb.spread_impact_dollars:+.2f}$ "
            f"timing={cb.timing_dollars:+.2f}$ "
            f"participation={cb.participation:.1%}"
        )
    if args.runs > 1:
        s = summarize(costs)
        print(
            f"\n{args.algo} over {int(s['runs'])} runs: IS {s['mean_is_bps']:.2f}±"
            f"{s['std_is_bps']:.2f} bps, CVaR95 {s['cvar95_is_bps']:.2f} bps, "
            f"completion {s['completion_rate']:.1%}, participation "
            f"{s['mean_participation']:.1%}"
        )


if __name__ == "__main__":
    main()
