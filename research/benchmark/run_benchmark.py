"""Benchmark-agent pipeline — full regime sweep + adversarial cells + selection.

Two passes of the shared harness:

  pass 1 (clean):  {exec, normal, volatile, thin, stressed} x registered algos
  pass 2 (attack): {exec, thin} x the four footprint adversaries combined

Results merge into one bench dict -> exec_scores -> write_leaderboard emits
LEADERBOARD.md + survivors.json next to this script. `rl` is excluded
(untrained at this commit); `learned` uses its predictor-free fallback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import orderflow.execution  # noqa: F401
from orderflow.eval.adversarial import (
    Frontrunner,
    LiquidityWithdrawer,
    MomentumIgniter,
    Spoofer,
)
from orderflow.eval.execution_bench import BenchConfig, run_bench
from orderflow.eval.select import write_leaderboard

OUT = Path(__file__).parent
ALGOS = ["twap", "twap_passive", "vwap", "ac"]  # rl untrained, learned needs a ckpt


def merge(a: dict, b: dict) -> dict:
    # merge per-regime tables: clean and adversarial cells coexist as
    # "algo" vs "algo@combined" keys in the same regime table
    for reg, table in b["regimes"].items():
        a["regimes"].setdefault(reg, {}).update(table)
    a["paired_vs_twap"].update(b["paired_vs_twap"])
    return a


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print("[bench] pass 1: clean regimes ...", flush=True)
    cfg_big = BenchConfig(
        n_episodes=2, seed=seed, tasks={"exec": (1800.0, 100_000)}, step=10.0
    )
    exec_res = run_bench(algos=ALGOS, regimes=["exec"], cfg=cfg_big)

    cfg_small = BenchConfig(
        n_episodes=4,
        seed=seed + 50,
        tasks={
            "normal": (600.0, 4_000),
            "volatile": (600.0, 3_000),
            "thin": (600.0, 800),
            "stressed": (600.0, 1_500),
        },
        step=10.0,
    )
    small_res = run_bench(
        algos=ALGOS, regimes=["normal", "volatile", "thin", "stressed"],
        cfg=cfg_small,
    )

    print("[bench] pass 2: combined adversaries on exec+thin ...", flush=True)
    adv = {"combined": (Spoofer, MomentumIgniter, LiquidityWithdrawer, Frontrunner)}
    exec_adv = run_bench(
        algos=ALGOS, regimes=["exec"], cfg=cfg_big, adversaries=adv
    )
    thin_adv = run_bench(
        algos=ALGOS, regimes=["thin"], cfg=cfg_small, adversaries=adv
    )

    bench = merge(merge(exec_res, small_res), merge(exec_adv, thin_adv))
    serializable = {
        "regimes": bench["regimes"],
        "paired_vs_twap": {f"{r}|{k}": v for (r, k), v in bench["paired_vs_twap"].items()},
    }
    (OUT / "results.json").write_text(json.dumps(serializable, indent=2))
    survivors = write_leaderboard(
        bench, {}, out_dir=OUT, md_path=OUT / "LEADERBOARD.md"
    )
    print(json.dumps(survivors["execution"], indent=2))
    print(f"[bench] wrote {OUT}/LEADERBOARD.md + survivors.json + results.json")


if __name__ == "__main__":
    main()
