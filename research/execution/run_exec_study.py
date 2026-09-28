"""Execution-research study — writes results.json next to this script.

Part A: head-to-head on the exec task ($10M notional, 30 min, 10 s grid)
across schedule algos with common random numbers — each algo sees the same
six market seeds so IS differences are paired, not environmental.

Part B: the Almgren–Chriss efficient frontier — E[C] vs sd(C) over a risk
aversion sweep — plus the *realized* IS of AC at three lambda settings to
check the closed-form frontier against what the sim actually charges.

Part C: learned vs static — the AdaptiveAC re-planner is the interesting
cell: does online vol re-solve beat fixed-kappa AC?
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import orderflow.execution  # noqa: F401 — populate the registry
from orderflow.book.types import Side
from orderflow.eval.execution_bench import run_episode
from orderflow.execution.almgren_chriss import (
    DEFAULT_ETA,
    DEFAULT_GAMMA,
    DEFAULT_SIGMA,
    ac_frontier,
)
from orderflow.execution.cost import summarize
from orderflow.execution.types import ExecutionTask
from orderflow.registry import get_execution
from orderflow.sim.regimes import get_regime

OUT = Path(__file__).parent / "results.json"
ALGOS = ["twap", "twap_passive", "twap_capped", "vwap", "ac", "ac_adaptive"]
SEEDS = [11, 12, 13, 14, 15, 16]


def run_study(seed_base: int) -> dict:
    flow_cls, params, seed_kwargs = get_regime("exec", flow="hawkes")
    task = ExecutionTask(side=Side.SELL, total_qty=100_000, horizon=1800.0, step=10.0)

    rows: dict[str, dict] = {}
    costs_by_algo: dict[str, list] = {}
    for name in ALGOS:
        t0 = time.time()
        costs = []
        for i, s in enumerate(SEEDS):
            algo = get_execution(name)()
            # common random numbers: same sim_seed for every algo at index i
            costs.append(
                run_episode(
                    algo, task, flow_cls, params, seed_kwargs,
                    sim_seed=seed_base * 100 + s,
                )
            )
        costs_by_algo[name] = costs
        summ = summarize(costs)
        summ["wall_seconds"] = round(time.time() - t0, 1)
        rows[name] = summ
        print(
            f"  {name:14s} mean IS {summ['mean_is_bps']:+.2f} bps  "
            f"std {summ['std_is_bps']:.2f}  cvar95 {summ['cvar95_is_bps']:.2f}  "
            f"fill {summ['mean_fill_rate']:.1%}  part {summ['mean_participation']:.1%}"
        )

    # paired diffs vs twap
    paired = {}
    for name in ALGOS:
        if name == "twap":
            continue
        d = np.array(
            [
                c.is_bps - t.is_bps
                for c, t in zip(costs_by_algo[name], costs_by_algo["twap"])
            ]
        )
        paired[f"{name} - twap"] = {
            "mean_diff_bps": float(d.mean()),
            "std_diff_bps": float(d.std()),
        }

    return {"algos": rows, "paired_vs_twap": paired}


def frontier_study() -> dict:
    lams = np.logspace(-7, -3, 9)
    e, s = ac_frontier(
        total_qty=100_000,
        horizon=1800.0,
        step=10.0,
        sigma=DEFAULT_SIGMA,
        eta=DEFAULT_ETA,
        gamma=DEFAULT_GAMMA,
        lambdas=lams,
    )
    return {
        "lambdas": lams.tolist(),
        "expected_cost_dollars": e.tolist(),
        "std_cost_dollars": s.tolist(),
        "params": {"sigma": DEFAULT_SIGMA, "eta": DEFAULT_ETA, "gamma": DEFAULT_GAMMA},
    }


def main() -> None:
    seed_base = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"[exec-study] seeds={SEEDS} algos={ALGOS}")
    print("[exec-study] part A: head-to-head ...", flush=True)
    study = run_study(seed_base)
    print("[exec-study] part B: AC frontier ...", flush=True)
    fr = frontier_study()
    results = {"seed_base": seed_base, "seeds": SEEDS, "study": study, "frontier": fr}
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[exec-study] wrote {OUT}")


if __name__ == "__main__":
    main()
