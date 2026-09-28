"""Bake demo fixtures for the GitHub Pages static build.

Runs real simulations/executions once and writes web/public/demo/*.json —
the VITE_DEMO build then serves these instead of a live API. Re-run this
script whenever the repo's headline results change.

    python scripts/bake_demo.py
    cd web && VITE_DEMO=1 npm run build   # -> ../docs
"""

from __future__ import annotations

import json
from pathlib import Path

import orderflow.execution  # noqa: F401
from orderflow.api.app import (
    ProbeRequest,
    SimRequest,
    StatsRequest,
    _run_probe,
    _run_sim,
    _run_stats,
    frontier,
)
from orderflow.book.types import Side
from orderflow.eval.adversarial import ADVERSARIES
from orderflow.eval.execution_bench import run_episode_traced
from orderflow.execution.cost import summarize as _summarize
from orderflow.execution.learned import LearnedPolicy, TorchSignal
from orderflow.execution.types import ExecutionTask
from orderflow.models.baselines import MLPBaseline
from orderflow.registry import get_execution, list_execution
from orderflow.sim.regimes import REGIMES, get_regime

OUT = Path("web/public/demo")
LAMS = (1e-6, 1e-5, 1e-4, 1e-3)


def dump(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj))
    print(f"  demo/{name}.json ({(OUT / (name + '.json')).stat().st_size // 1024} KB)")


def bake_exec(name: str, algo_name: str, adv_names: tuple[str, ...], seed: int = 5) -> None:
    flow_cls, params, seed_kwargs = get_regime("exec", flow="hawkes")
    side = Side.SELL
    task = ExecutionTask(side=side, total_qty=100_000, horizon=1800.0, step=10.0)
    if algo_name == "learned":
        algo = LearnedPolicy(
            predictor=TorchSignal(MLPBaseline.load("artifacts/models/mlp_mid_move.pt"), side)
        )
    else:
        algo = get_execution(algo_name)()
    advs = tuple(ADVERSARIES[k]() for k in adv_names)
    cb, trace, extra = run_episode_traced(
        algo, task, flow_cls, params, seed_kwargs, sim_seed=seed, adversaries=advs
    )
    s = _summarize([cb])
    dump(
        name,
        {
            "algo": algo_name,
            "adversaries": list(adv_names),
            "anatomy": {"trace": trace, **extra},
            "episodes": [
                {
                    "seed": seed,
                    "filled": cb.filled_qty,
                    "is_bps": round(cb.is_bps, 3),
                    "is_dollars": round(cb.is_dollars, 2),
                    "spread_impact": round(cb.spread_impact_dollars, 2),
                    "timing": round(cb.timing_dollars, 2),
                    "participation": round(cb.participation, 4),
                }
            ],
            "summary": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
            },
        },
    )


def main() -> None:
    print("[bake] catalog + leaderboard + models")
    dump(
        "catalog",
        {
            "regimes": sorted(REGIMES),
            "flows": ["hawkes", "zi", "queue_reactive"],
            "algos": list_execution(),
            "adversaries": sorted(ADVERSARIES),
        },
    )
    lb, sv = Path("LEADERBOARD.md"), Path("survivors.json")
    dump(
        "leaderboard",
        {
            "leaderboard_md": lb.read_text() if lb.exists() else None,
            "survivors": json.loads(sv.read_text()) if sv.exists() else None,
        },
    )
    pred = Path("research/benchmark/pred_results.json")
    dump(
        "models",
        {"pred_bench": json.loads(pred.read_text()) if pred.exists() else None},
    )

    print("[bake] frontier (curve + schedules at a few lambdas)")
    base = frontier()
    schedules = {f"{lam:g}": frontier(lam=lam)["at_lam"] for lam in LAMS}
    dump(
        "frontier",
        {
            "curve": {
                "lambdas": base["lambdas"],
                "expected": base["expected"],
                "std": base["std"],
            },
            "schedules": schedules,
            "params": base["params"],
        },
    )

    print("[bake] sim (normal/hawkes 120s)")
    dump("sim", _run_sim(SimRequest(regime="normal", flow="hawkes", horizon=120, seed=11)))

    print("[bake] stats (normal/hawkes 600s)")
    dump("stats", _run_stats(StatsRequest(regime="normal", flow="hawkes", horizon=600, seed=3)))

    print("[bake] exec: ac clean on exec regime")
    bake_exec("exec_ac", "ac", ())
    print("[bake] exec: twap_passive under combined attack")
    bake_exec("exec_twap_passive_adv", "twap_passive", ("withdrawer", "spoofer", "igniter"))

    print("[bake] probe (normal 120s, 24 rows)")
    dump(
        "probe",
        _run_probe(ProbeRequest(regime="normal", flow="hawkes", horizon=120, n_probe=24, seed=5)),
    )
    print("[bake] done")


if __name__ == "__main__":
    main()
