"""Survival rule + leaderboard generation.

Execution algos are ranked by a robust score

    S = mean_IS_bps + 0.5 * CVaR95_IS_bps    (averaged over regimes)

and an algo *survives* only if its completion rate is >= 0.999 in every regime
it ran — an algo that leaves shares on the table at the deadline fails the task
regardless of cost. Predictors survive if they beat the Markov baseline in NLL
on at least 4 of the 6 evaluated targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRED_TARGETS = (
    "next_type",
    "mid_move",
    "spread_change",
    "cancel_prob",
    "short_vol",
    "queue_depletion",
)


def exec_scores(bench: dict) -> dict[str, dict]:
    """Per-algo robust score averaged over regimes (and per-regime table)."""
    regimes = bench.get("regimes", {})
    algos = sorted({a for t in regimes.values() for a in t})
    out: dict[str, dict] = {}
    for a in algos:
        scores, completes, rows = [], [], {}
        for regime, table in regimes.items():
            if a not in table:
                continue
            s = table[a]
            rows[regime] = s
            scores.append(s["mean_is_bps"] + 0.5 * s["cvar95_is_bps"])
            completes.append(s["completion_rate"] >= 0.999)
        if not scores:
            continue
        out[a] = {
            "score": float(np.mean(scores)),
            "survives": bool(all(completes)),
            "regimes": rows,
        }
    return out


def pred_scores(pred_bench: dict, baseline: str = "markov") -> dict[str, dict]:
    """Predictor survival: beat `baseline` NLL on >= 4/6 targets, per regime."""
    out: dict[str, dict] = {}
    for regime, table in pred_bench.items():
        base = table.get(baseline, {})
        for name, metrics in table.items():
            if name == baseline:
                continue
            wins = 0
            detail = {}
            for t in PRED_TARGETS:
                k = f"{t}_nll"
                win = metrics.get(k, np.inf) < base.get(k, np.inf)
                wins += bool(win)
                detail[t] = {"nll": metrics.get(k), "beat_baseline": bool(win)}
            rec = out.setdefault(name, {"wins": 0, "targets": 0, "regimes": {}})
            rec["wins"] += wins
            rec["targets"] += len(PRED_TARGETS)
            rec["regimes"][regime] = detail
    for name, rec in out.items():
        rec["survives"] = rec["targets"] > 0 and rec["wins"] >= 4 * len(rec["regimes"])
    return out


def write_leaderboard(
    exec_bench: dict,
    pred_bench: dict,
    out_dir: str | Path = ".",
    md_path: str | Path = "LEADERBOARD.md",
) -> dict:
    """Write survivors.json + a human-readable LEADERBOARD.md."""
    es = exec_scores(exec_bench)
    ps = pred_scores(pred_bench)
    survivors = {
        "execution": {a: v["survives"] for a, v in es.items()},
        "prediction": {a: v["survives"] for a, v in ps.items()},
        "scores_exec": {a: v["score"] for a, v in es.items()},
        "wins_pred": {a: f"{v['wins']}/{v['targets']}" for a, v in ps.items()},
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "survivors.json").write_text(json.dumps(survivors, indent=2))

    lines = ["# OrderFlow Leaderboard", ""]
    lines += [
        "Execution score `S = mean_IS_bps + 0.5 * CVaR95` (lower is better),",
        "averaged over regimes. An algo survives only with completion >= 99.9%",
        "in every regime. Predictors survive by beating the Markov baseline in",
        "NLL on at least 4/6 targets per regime.",
        "",
        "## Execution (IS bps, mean +/- std | CVaR95 | completion)",
        "",
    ]
    for a, v in sorted(es.items(), key=lambda kv: kv[1]["score"]):
        flag = "SURVIVES" if v["survives"] else "eliminated"
        lines.append(f"- **{a}** — score {v['score']:.2f} ({flag})")
        for reg, s in v["regimes"].items():
            lines.append(
                f"  - {reg}: {s['mean_is_bps']:+.2f} ± {s['std_is_bps']:.2f} | "
                f"CVaR {s['cvar95_is_bps']:+.2f} | done {s['completion_rate']:.0%} | "
                f"part {s['mean_participation']:.0%}"
            )
    lines += ["", "## Prediction (targets beaten vs Markov baseline)", ""]
    for a, v in ps.items():
        flag = "SURVIVES" if v["survives"] else "eliminated"
        lines.append(f"- **{a}** — {v['wins']}/{v['targets']} target wins ({flag})")
    Path(md_path).write_text("\n".join(lines) + "\n")
    return survivors
