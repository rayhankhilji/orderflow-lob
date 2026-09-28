"""OrderFlow API — FastAPI surface for the simulator, execution, and results.

Run:  uvicorn orderflow.api.app:app --reload --port 8000

All heavy work goes through the in-memory job runner (`POST` returns a job
id; `GET /api/jobs/{id}` polls status + result). State is process-local by
design — this is a research tool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import orderflow.execution  # noqa: F401 — register algos
from orderflow.api import jobs
from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.eval.execution_bench import run_episode
from orderflow.execution.cost import summarize
from orderflow.execution.types import ExecutionTask
from orderflow.registry import get_execution, list_execution
from orderflow.sim.regimes import REGIMES, get_regime
from orderflow.sim.simulator import Simulator, seed_book

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
    costs = []
    for i in range(req.n_episodes):
        cb = run_episode(
            _make_algo(req, side),
            task,
            flow_cls,
            params,
            seed_kwargs,
            sim_seed=req.seed + i,
        )
        costs.append(cb)
    s = summarize(costs)
    return {
        "algo": req.algo,
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
        import json

        out["survivors"] = json.loads(sv.read_text())
    return out
