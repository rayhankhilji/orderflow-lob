"""OrderFlow API — FastAPI surface for the simulator, execution, and results.

Run:  uvicorn orderflow.api.app:app --reload --port 8000

All heavy work goes through the in-memory job runner (`POST` returns a job
id; `GET /api/jobs/{id}` polls status + result). State is process-local by
design — this is a research tool.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import orderflow.execution  # noqa: F401 — register algos
from orderflow.api import jobs
from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.eval.adversarial import ADVERSARIES
from orderflow.eval.execution_bench import run_episode, run_episode_traced
from orderflow.execution.almgren_chriss import (
    DEFAULT_ETA,
    DEFAULT_GAMMA,
    DEFAULT_SIGMA,
    ac_cost_variance,
    ac_expected_cost,
    ac_frontier,
    ac_inventory,
    ac_kappa,
)
from orderflow.execution.cost import summarize
from orderflow.execution.types import ExecutionTask
from orderflow.registry import get_execution, list_execution
from orderflow.sim.regimes import REGIMES, get_regime
from orderflow.sim.simulator import Simulator, seed_book
from orderflow.stats.impact import kyle_lambda
from orderflow.stats.ofi import compute_ofi_series, ofi_regression
from orderflow.stats.stylized import stylized_facts

app = FastAPI(title="OrderFlow", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class SimRequest(BaseModel):
    regime: str = "normal"
    flow: str = "hawkes"
    horizon: float = Field(300.0, gt=0, le=3600)
    seed: int = 0
    snapshot_every: float = Field(1.0, gt=0.05)


class ExecRequest(BaseModel):
    algo: str = "twap"
    regime: str = "exec"
    side: str = "SELL"
    total_qty: int = Field(100_000, gt=0)
    horizon: float = Field(1800.0, gt=0, le=7200)
    step: float = Field(10.0, gt=0)
    n_episodes: int = Field(2, ge=1, le=10)
    seed: int = 0
    adversaries: list[str] = []
    trace: bool = True


class StatsRequest(BaseModel):
    regime: str = "normal"
    flow: str = "hawkes"
    horizon: float = Field(300.0, gt=10, le=1800)
    seed: int = 0


class ProbeRequest(BaseModel):
    regime: str = "normal"
    flow: str = "hawkes"
    horizon: float = Field(120.0, gt=10, le=600)
    seed: int = 0
    n_probe: int = Field(24, ge=4, le=64)


def _event_rows(res, limit: int = 400) -> list[dict]:
    evs = res.events[-limit:]
    return [
        {
            "t": round(float(e.timestamp), 3),
            "type": e.type.value,
            "side": int(e.side),
            "price": e.price,
            "qty": e.qty,
            "market": bool(e.is_market),
        }
        for e in evs
    ]


def _depth_rows(snapshot) -> dict | None:
    if snapshot is None or len(snapshot.bid_prices) == 0:
        return None
    return {
        "t": float(snapshot.timestamp),
        "bid_prices": [int(p) for p in snapshot.bid_prices],
        "bid_qtys": [int(q) for q in snapshot.bid_qtys],
        "ask_prices": [int(p) for p in snapshot.ask_prices],
        "ask_qtys": [int(q) for q in snapshot.ask_qtys],
    }


def _run_sim(req: SimRequest) -> dict:
    flow_cls, params, seed_kwargs = get_regime(req.regime, flow=req.flow)
    rng = np.random.default_rng(req.seed)
    book = LimitOrderBook(tick_size=0.01)
    seed_book(book, rng=rng, **seed_kwargs)
    sim = Simulator(book, flow_cls(params), seed=req.seed)
    res = sim.run(req.horizon, snapshot_every=req.snapshot_every)

    # mid series from snapshots (uniform grid) — res.mid_series is event-driven
    # and too sparse early in a run to draw a useful chart
    mids = [
        (float(s.timestamp), (s.bid_prices[0] + s.ask_prices[0]) / 2)
        for s in res.snapshots
        if len(s.bid_prices) and len(s.ask_prices)
    ]
    step = max(1, len(mids) // 1500)
    mid_ds = [[round(t, 2), round(m * 0.01, 4)] for t, m in mids[::step]]
    spreads = [
        s.ask_prices[0] - s.bid_prices[0]
        for s in res.snapshots
        if s.bid_prices[0] and s.ask_prices[0]
    ]
    return {
        "summary": {
            "events": len(res.events),
            "fills": len(res.fills),
            "events_per_sec": round(len(res.events) / req.horizon, 2),
            "mean_spread_ticks": (
                round(float(np.mean(spreads)), 3) if spreads else None
            ),
            "horizon": req.horizon,
            "regime": req.regime,
            "flow": req.flow,
        },
        "mid_series": mid_ds,
        "tape": _event_rows(res),
        "depth": _depth_rows(res.snapshots[-1] if res.snapshots else None),
        "event_counts": {k: len(v) for k, v in res.event_times_by_type.items()},
    }


_MODELS_DIR = Path("artifacts/models")


def _make_algo(req: ExecRequest, side: Side):
    """Build the algo, attaching trained checkpoints when available.

    `learned` needs artifacts/models/mlp_mid_move.pt and `rl` needs
    artifacts/models/ppo_exec.pt (both written by scripts/final_eval.py).
    Without them they degrade to plain AC / TWAP-slice behavior.
    """
    if req.algo == "learned" and (_MODELS_DIR / "mlp_mid_move.pt").exists():
        from orderflow.execution.learned import LearnedPolicy, TorchSignal
        from orderflow.models.baselines import MLPBaseline

        return LearnedPolicy(
            predictor=TorchSignal(MLPBaseline.load(_MODELS_DIR / "mlp_mid_move.pt"), side)
        )
    if req.algo == "rl" and (_MODELS_DIR / "ppo_exec.pt").exists():
        from orderflow.execution.rl.policy import RLExecution

        return RLExecution().load_policy(str(_MODELS_DIR / "ppo_exec.pt"))
    return get_execution(req.algo)()


def _run_exec(req: ExecRequest) -> dict:
    flow_cls, params, seed_kwargs = get_regime(req.regime, flow="hawkes")
    side = Side.SELL if req.side.upper() == "SELL" else Side.BUY
    task = ExecutionTask(
        side=side, total_qty=req.total_qty, horizon=req.horizon, step=req.step
    )
    unknown = [a for a in req.adversaries if a not in ADVERSARIES]
    if unknown:
        raise ValueError(f"unknown adversaries: {unknown}")
    costs = []
    anatomy = None
    for i in range(req.n_episodes):
        adv = tuple(ADVERSARIES[k]() for k in req.adversaries)
        if req.trace and i == 0:
            cb, trace, extra = run_episode_traced(
                _make_algo(req, side),
                task,
                flow_cls,
                params,
                seed_kwargs,
                sim_seed=req.seed + i,
                adversaries=adv,
            )
            anatomy = {"trace": trace, **extra}
        else:
            cb = run_episode(
                _make_algo(req, side),
                task,
                flow_cls,
                params,
                seed_kwargs,
                sim_seed=req.seed + i,
                adversaries=adv,
            )
        costs.append(cb)
    s = summarize(costs)
    return {
        "algo": req.algo,
        "adversaries": req.adversaries,
        "anatomy": anatomy,
        "episodes": [
            {
                "seed": req.seed + i,
                "filled": c.filled_qty,
                "is_bps": round(c.is_bps, 3),
                "is_dollars": round(c.is_dollars, 2),
                "spread_impact": round(c.spread_impact_dollars, 2),
                "timing": round(c.timing_dollars, 2),
                "participation": (
                    round(c.participation, 4)
                    if np.isfinite(c.participation)
                    else None
                ),
            }
            for i, c in enumerate(costs)
        ],
        "summary": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()},
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/regimes")
def regimes() -> dict:
    return {"regimes": sorted(REGIMES), "flows": ["hawkes", "zi", "queue_reactive"]}


@app.get("/api/algos")
def algos() -> dict:
    return {"algos": list_execution()}


@app.post("/api/sim")
def run_sim(req: SimRequest) -> dict:
    if req.regime not in REGIMES:
        raise HTTPException(400, f"unknown regime {req.regime!r}")
    job = jobs.submit("sim", lambda: _run_sim(req))
    return {"job_id": job.id}


@app.post("/api/exec")
def run_exec(req: ExecRequest) -> dict:
    if req.algo not in list_execution():
        raise HTTPException(400, f"unknown algo {req.algo!r}")
    job = jobs.submit("exec", lambda: _run_exec(req))
    return {"job_id": job.id}


@app.get("/api/jobs")
def job_list() -> dict:
    return {"jobs": [jobs.public(j) for j in jobs.all_jobs()[:50]]}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: int) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return jobs.public(job)


@app.get("/api/leaderboard")
def leaderboard() -> dict:
    lb = REPO_ROOT / "LEADERBOARD.md"
    sv = REPO_ROOT / "survivors.json"
    out = {"leaderboard_md": None, "survivors": None}
    if lb.exists():
        out["leaderboard_md"] = lb.read_text()
    if sv.exists():
        out["survivors"] = json.loads(sv.read_text())
    return out


@app.get("/api/adversaries")
def adversaries() -> dict:
    return {"adversaries": sorted(ADVERSARIES)}


@app.get("/api/frontier")
def frontier(
    total_qty: int = 100_000,
    horizon: float = 1800.0,
    step: float = 10.0,
    lam: float = 1e-4,
    sigma: float = DEFAULT_SIGMA,
    eta: float = DEFAULT_ETA,
    gamma: float = DEFAULT_GAMMA,
) -> dict:
    """Almgren-Chriss efficient frontier + the schedule at a chosen lambda.

    Pure math endpoint (no simulation): E[C] and std[C] over a lambda grid,
    plus the optimal inventory trajectory for the requested lambda.
    """
    if total_qty <= 0 or horizon <= 0 or step <= 0:
        raise HTTPException(400, "qty/horizon/step must be positive")
    lambdas = np.logspace(-7, -3, 23)
    e, s = ac_frontier(total_qty, horizon, step, sigma, eta, gamma, lambdas)
    kappa = ac_kappa(sigma, eta, gamma, lam, step)
    inv = ac_inventory(total_qty, horizon, step, kappa)
    ts = np.arange(len(inv)) * step
    return {
        "lambdas": [round(float(l), 9) for l in lambdas],
        "expected": [round(float(v), 1) for v in e],
        "std": [round(float(v), 1) for v in s],
        "at_lam": {
            "lambda": lam,
            "kappa": round(float(kappa), 6),
            "expected": round(float(ac_expected_cost(inv, step, sigma, eta, gamma)), 1),
            "std": round(float(math.sqrt(ac_cost_variance(inv, step, sigma))), 1),
            "schedule_t": [round(float(t), 1) for t in ts],
            "schedule_x": [round(float(x), 1) for x in inv],
        },
        "params": {"sigma": sigma, "eta": eta, "gamma": gamma},
    }


def _run_stats(req: StatsRequest) -> dict:
    flow_cls, params, seed_kwargs = get_regime(req.regime, flow=req.flow)
    rng = np.random.default_rng(req.seed)
    book = LimitOrderBook(tick_size=0.01)
    seed_book(book, rng=rng, **seed_kwargs)
    sim = Simulator(book, flow_cls(params), seed=req.seed)
    res = sim.run(req.horizon, snapshot_every=1.0)

    ofi = compute_ofi_series(res.snapshots)
    mids = np.array(
        [
            (s.bid_prices[0] + s.ask_prices[0]) / 2
            if len(s.bid_prices) and len(s.ask_prices)
            else np.nan
            for s in res.snapshots
        ]
    )
    dm = np.diff(mids, prepend=mids[0])
    fit = ofi_regression(ofi, dm)
    fit_w = ofi_regression(ofi, dm, window=10)
    mask = np.isfinite(ofi) & np.isfinite(dm)
    step = max(1, int(mask.sum() // 800))

    facts = stylized_facts(res)
    # Kyle lambda: signed taker volume vs mid change in 5 s buckets
    kyle = None
    if res.fills:
        t_end = res.fills[-1].timestamp
        nb = max(1, int(t_end // 5))
        vol = np.zeros(nb)
        for f in res.fills:
            vol[min(int(f.timestamp // 5), nb - 1)] += f.taker_side.value * f.qty
        ts = np.array([s.timestamp for s in res.snapshots])
        edges = np.arange(nb + 1) * 5.0
        idx = np.clip(np.searchsorted(ts, edges) - 1, 0, len(ts) - 1)
        dp = np.diff(mids[idx])
        ok = np.isfinite(dp) & np.isfinite(vol[: len(dp)])
        if ok.sum() > 8:
            kyle = kyle_lambda(vol[: len(dp)][ok], dp[ok]) * 1000.0

    return {
        "summary": {
            "events": len(res.events),
            "fills": len(res.fills),
            "regime": req.regime,
            "flow": req.flow,
            "kyle_lambda_ticks_per_1k": (round(kyle, 3) if kyle is not None else None),
        },
        "ofi": {
            "beta": round(float(fit.beta), 5),
            "r2": round(float(fit.r2), 4),
            "t_stat": round(float(fit.t_stat), 2),
            "n_obs": int(fit.n_obs),
            "windowed_r2": round(float(fit_w.r2), 4),
            "windowed_t": round(float(fit_w.t_stat), 2),
            "scatter": [
                [round(float(x), 1), round(float(y), 2)]
                for x, y in zip(ofi[mask][::step], dm[mask][::step])
            ],
        },
        "moments": {k: round(float(v), 4) for k, v in facts["moments"].items()},
        "absret_acf": [round(v, 4) for v in facts["absret_acf"][:40]],
        "sign_acf": [round(v, 4) for v in facts["sign_acf"][:40]],
    }


@app.post("/api/stats")
def run_stats(req: StatsRequest) -> dict:
    if req.regime not in REGIMES:
        raise HTTPException(400, f"unknown regime {req.regime!r}")
    job = jobs.submit("stats", lambda: _run_stats(req))
    return {"job_id": job.id}


@app.get("/api/models")
def models() -> dict:
    pred = REPO_ROOT / "research/benchmark/pred_results.json"
    out = {"pred_bench": None, "targets": None}
    if pred.exists():
        out["pred_bench"] = json.loads(pred.read_text())
    return out


def _run_probe(req: ProbeRequest) -> dict:
    """Run a fresh episode and ask the trained MLP what it expects next."""
    from orderflow.models.baselines import MLPBaseline
    from orderflow.models.dataset import TARGET_KEYS
    from orderflow.models.features import tokenize
    from orderflow.models.targets import make_targets

    ck_path = _MODELS_DIR / "mlp_mid_move.pt"
    if not ck_path.exists():
        raise ValueError(
            "no trained checkpoint — run scripts/final_eval.py to produce artifacts/models/"
        )
    flow_cls, params, seed_kwargs = get_regime(req.regime, flow=req.flow)
    rng = np.random.default_rng(req.seed)
    book = LimitOrderBook(tick_size=0.01)
    seed_book(book, rng=rng, **seed_kwargs)
    sim = Simulator(
        book, flow_cls(params), seed=req.seed, record_state_every_event=True
    )
    res = sim.run(req.horizon)

    stream = tokenize(res)
    targets = make_targets(stream, res)
    mlp = MLPBaseline.load(ck_path)

    n = len(stream)
    valid = np.asarray(targets["valid"])
    mid_move = np.asarray(targets["mid_move"])
    pool = np.where(valid)[0]
    take = pool[
        np.linspace(0, len(pool) - 1, min(req.n_probe, len(pool))).astype(int)
    ]
    X = torch.tensor(stream.state[take][:, None, :], dtype=torch.float32)
    probs = mlp.predict({"state": X})["mid_move"][:, 0, :]  # (K, 3)

    labels = ["down", "flat", "up"]
    rows = []
    ev = res.flow_events
    for j, i in enumerate(take):
        rows.append(
            {
                "event_idx": int(i),
                "t": round(float(ev[i].timestamp), 2),
                "event_type": ev[i].type.value,
                "pred": {
                    labels[k]: round(float(probs[j, k]), 3) for k in range(3)
                },
                "realized": labels[int(mid_move[i])],
                "correct": int(np.argmax(probs[j])) == int(mid_move[i]),
            }
        )
    acc = float(np.mean([r["correct"] for r in rows]))
    return {
        "n_events": n,
        "checkpoint": "mlp_mid_move.pt",
        "targets_available": list(TARGET_KEYS),
        "probe_accuracy": round(acc, 3),
        "rows": rows,
    }


@app.post("/api/probe")
def probe(req: ProbeRequest) -> dict:
    if req.regime not in REGIMES:
        raise HTTPException(400, f"unknown regime {req.regime!r}")
    job = jobs.submit("probe", lambda: _run_probe(req))
    return {"job_id": job.id}
