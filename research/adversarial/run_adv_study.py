"""Adversarial-research study — attack uplift matrix.

Each adversary is paired with each algo on identical sim seeds; the metric
is IS uplift = mean(IS_with_adversary) - mean(IS_alone), in bps. A positive
uplift means the adversary made execution more expensive. Runs the scaled
task (20k shares / 600 s) so the full matrix stays cheap. Writes
results.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import orderflow.execution  # noqa: F401
from orderflow.book.types import Side
from orderflow.eval.adversarial import ADVERSARIES
from orderflow.eval.execution_bench import run_episode
from orderflow.execution.types import ExecutionTask
from orderflow.registry import get_execution
from orderflow.sim.regimes import get_regime

OUT = Path(__file__).parent / "results.json"
QTY, HORIZON, STEP = 20_000, 600.0, 10.0
ALGOS = ["twap", "twap_passive", "ac"]
ADVERSARY_SETS = {
    "none": (),
    "spoofer": ("spoofer",),
    "igniter": ("igniter",),
    "withdrawer": ("withdrawer",),
    "frontrunner": ("frontrunner",),
    "pennyjumper": ("pennyjumper",),
    "combined": ("spoofer", "igniter", "withdrawer", "pennyjumper"),
}
SEEDS = [51, 52, 53, 54]


def is_stats(costs) -> dict:
    isb = [c.is_bps for c in costs]
    return {
        "mean_is_bps": float(sum(isb) / len(isb)),
        "fill_rate": float(sum(c.fill_rate for c in costs) / len(costs)),
    }


def main() -> None:
    seed_base = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    flow_cls, params, seed_kwargs = get_regime("exec", flow="hawkes")
    task = ExecutionTask(side=Side.SELL, total_qty=QTY, horizon=HORIZON, step=STEP)

    matrix: dict[str, dict[str, dict]] = {}
    for algo_name in ALGOS:
        matrix[algo_name] = {}
        for adv_name, adv_keys in ADVERSARY_SETS.items():
            t0 = time.time()
            costs = []
            for s in SEEDS:
                # fresh adversary instances per episode (they hold state)
                adv = tuple(ADVERSARIES[k]() for k in adv_keys)
                costs.append(
                    run_episode(
                        get_execution(algo_name)(),
                        task,
                        flow_cls,
                        params,
                        seed_kwargs,
                        sim_seed=seed_base * 100 + s,
                        adversaries=adv,
                    )
                )
            matrix[algo_name][adv_name] = is_stats(costs)
            print(
                f"  {algo_name:14s} vs {adv_name:11s} "
                f"IS {matrix[algo_name][adv_name]['mean_is_bps']:+7.2f} bps "
                f"fill {matrix[algo_name][adv_name]['fill_rate']:.1%} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )

    uplift = {
        a: {
            adv: matrix[a][adv]["mean_is_bps"] - matrix[a]["none"]["mean_is_bps"]
            for adv in ADVERSARY_SETS
            if adv != "none"
        }
        for a in ALGOS
    }
    results = {
        "task": [QTY, HORIZON, STEP],
        "seeds": SEEDS,
        "matrix": matrix,
        "uplift_bps": uplift,
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[adv-study] wrote {OUT}")


if __name__ == "__main__":
    main()
