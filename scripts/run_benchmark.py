"""Run the benchmark suite and write LEADERBOARD.md + artifacts/eval/.

Usage:
    python scripts/run_benchmark.py --bench execution --episodes 4
    python scripts/run_benchmark.py --bench prediction --regimes normal
    python scripts/run_benchmark.py --bench all --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from orderflow.eval.adversarial import ADVERSARIES
from orderflow.eval.execution_bench import BenchConfig, run_bench
from orderflow.eval.prediction_bench import PredBenchConfig, run_pred_bench
from orderflow.eval.select import write_leaderboard


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="all", choices=["execution", "prediction", "all"])
    ap.add_argument("--flow", default="hawkes")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--algos", default=None, help="comma list; default all non-rl")
    ap.add_argument("--regimes", default=None)
    ap.add_argument("--adversarial", action="store_true",
                    help="also run each algo under each adversary")
    ap.add_argument("--quick", action="store_true",
                    help="tiny config: normal regime, 2 episodes, 120s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/eval")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    exec_res, pred_res = {"regimes": {}}, {}
    if args.bench in ("execution", "all"):
        cfg = BenchConfig(flow=args.flow, seed=args.seed)
        if args.quick:
            cfg.tasks = {"normal": (120.0, 500)}
            cfg.n_episodes = 2
        else:
            cfg.n_episodes = args.episodes
        algos = args.algos.split(",") if args.algos else None
        regimes = args.regimes.split(",") if args.regimes else None
        adversaries = {"none": ()}
        if args.adversarial:
            adversaries.update({k: (v,) for k, v in ADVERSARIES.items()})
        exec_res = run_bench(algos=algos, regimes=regimes, cfg=cfg,
                             adversaries=adversaries)
        (out / "execution_bench.json").write_text(
            json.dumps(_jsonable(exec_res), indent=2)
        )

    if args.bench in ("prediction", "all"):
        pcfg = PredBenchConfig(flow=args.flow, seed=args.seed)
        if args.quick:
            pcfg.n_train, pcfg.n_test, pcfg.horizon = 3, 1, 120.0
        regimes = args.regimes.split(",") if args.regimes else ["normal"]
        pred_res = run_pred_bench(regimes=regimes, cfg=pcfg)
        (out / "prediction_bench.json").write_text(
            json.dumps(_jsonable(pred_res), indent=2)
        )

    surv = write_leaderboard(exec_res, pred_res, out_dir=out, md_path="LEADERBOARD.md")
    print(f"\nsurvivors: {json.dumps(surv['execution'])} "
          f"exec / {json.dumps(surv['prediction'])} pred")
    print(f"wrote {out}/ and LEADERBOARD.md in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
