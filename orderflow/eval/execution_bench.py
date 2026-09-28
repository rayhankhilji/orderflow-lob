"""Execution benchmark: every registered algo x every regime x N seeds.

Common random numbers: episode (algo, regime, i) uses seed = base_seed + i for
the flow/book RNG regardless of algo, so paired comparisons vs TWAP are on
identical market paths (algo orders change downstream state, but the underlying
draws are shared).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import orderflow.execution  # noqa: F401 - populates the registry
from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Order, Side
from orderflow.execution.base import AlgoParticipant
from orderflow.execution.cost import CostBreakdown, cost_breakdown, summarize
from orderflow.execution.types import ExecutionTask
from orderflow.registry import get_execution, list_execution
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


@dataclass
class BenchConfig:
    flow: str = "hawkes"
    n_episodes: int = 4
    side: Side = Side.SELL
    # per-regime (horizon, qty) — exec is the flagship $10M-scale task,
    # others scale the qty down to their thinner books
    tasks: dict = field(
        default_factory=lambda: {
            "exec": (1800.0, 100_000),
            "normal": (600.0, 4_000),
            "volatile": (600.0, 3_000),
            "thin": (600.0, 800),
        }
    )
    step: float = 10.0
    tick_size: float = 0.01
    seed: int = 0
    n_bootstrap: int = 500


def run_episode(
    algo,
    task: ExecutionTask,
    flow_cls,
    params: dict,
    seed_kwargs: dict,
    sim_seed: int,
    tick_size: float = 0.01,
    adversaries: tuple = (),
) -> CostBreakdown:
    """One seeded execution episode; returns the cost decomposition."""
    rng = np.random.default_rng(sim_seed)
    book = LimitOrderBook(tick_size=tick_size)
    seed_book(book, rng=rng, **seed_kwargs)
    algo.reset(task, book, np.random.default_rng(sim_seed + 7919))
    adapter = AlgoParticipant(algo, task)
    sim = Simulator(
        book,
        flow_cls(dict(params)),
        seed=sim_seed,
        participants=[adapter, *adversaries],
    )
    res = sim.run(task.horizon + 1.0)
    arrival = adapter._arrival_mid or 0.0
    return cost_breakdown(
        side=task.side,
        requested_qty=task.total_qty,
        fills=adapter._fills,
        mid_series=res.mid_series,
        arrival_mid_ticks=arrival,
        tick_size=tick_size,
        total_sim_volume=float(sum(f.qty for f in res.fills)),
    )


class _Tracer(AlgoParticipant):
    """AlgoParticipant that records a per-decision trace for the UI.

    Rows are appended at each decision wake-up *after* fills are collected:
    time, mid, remaining inventory, cumulative realized IS so far (approx —
    recomputed exactly at the end by cost_breakdown), and the child orders
    the algo emitted at this step.
    """

    def __init__(self, algo, task, tick_size: float = 0.01) -> None:
        super().__init__(algo, task)
        self.tick_size = tick_size
        self.trace: list[dict] = []

    def on_time(self, book, t):
        reqs = super().on_time(book, t)
        mid = book.mid()
        children = [o for o in reqs if isinstance(o, Order)]
        self.trace.append(
            {
                "t": round(float(t), 2),
                "mid": (round(float(mid) * self.tick_size, 4) if mid is not None else None),
                "remaining": int(self.remaining),
                "filled": int(sum(f.qty for f in self._fills)),
                "children": [
                    {
                        "kind": o.order_type.value,
                        "qty": int(o.qty),
                        "price": (
                            round(float(o.price) * self.tick_size, 4)
                            if o.price is not None
                            else None
                        ),
                    }
                    for o in children
                ],
            }
        )
        return reqs


def run_episode_traced(
    algo,
    task: ExecutionTask,
    flow_cls,
    params: dict,
    seed_kwargs: dict,
    sim_seed: int,
    tick_size: float = 0.01,
    adversaries: tuple = (),
) -> tuple[CostBreakdown, list[dict], list[dict]]:
    """run_episode + a decision trace and exec-fill marks for the UI."""
    rng = np.random.default_rng(sim_seed)
    book = LimitOrderBook(tick_size=tick_size)
    seed_book(book, rng=rng, **seed_kwargs)
    algo.reset(task, book, np.random.default_rng(sim_seed + 7919))
    adapter = _Tracer(algo, task, tick_size=tick_size)
    sim = Simulator(
        book,
        flow_cls(dict(params)),
        seed=sim_seed,
        participants=[adapter, *adversaries],
    )
    res = sim.run(task.horizon + 1.0)
    arrival = adapter._arrival_mid or 0.0
    cb = cost_breakdown(
        side=task.side,
        requested_qty=task.total_qty,
        fills=adapter._fills,
        mid_series=res.mid_series,
        arrival_mid_ticks=arrival,
        tick_size=tick_size,
        total_sim_volume=float(sum(f.qty for f in res.fills)),
    )
    fill_marks = [
        {
            "t": round(float(f.timestamp), 2),
            "price": round(float(f.price) * tick_size, 4),
            "qty": int(f.qty),
        }
        for f in adapter._fills
    ]
    mids = [
        (round(float(t), 2), round(float(m) * tick_size, 4))
        for t, m in res.mid_series
    ]
    step = max(1, len(mids) // 1500)
    return cb, adapter.trace, {"mid_series": mids[::step], "fills": fill_marks}


def paired_bootstrap(
    diffs: np.ndarray, n_boot: int = 500, seed: int = 0
) -> tuple[float, float, float]:
    """Mean paired diff and a 95% bootstrap CI."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return (
        float(np.mean(d)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def run_bench(
    algos: list[str] | None = None,
    regimes: list[str] | None = None,
    cfg: BenchConfig | None = None,
    adversaries: dict[str, tuple] | None = None,
    verbose: bool = True,
) -> dict:
    """Full benchmark. Returns {regime: {algo: summary_dict}} plus paired rows."""
    cfg = cfg or BenchConfig()
    algos = algos or [a for a in list_execution() if a != "rl"] or list_execution()
    regimes = regimes or list(cfg.tasks)
    adversaries = adversaries or {"none": ()}
    out: dict = {"regimes": {}, "paired_vs_twap": {}}
    for regime in regimes:
        if regime not in cfg.tasks:
            continue
        horizon, qty = cfg.tasks[regime]
        flow_cls, params, seed_kwargs = get_regime(regime, flow=cfg.flow)
        table: dict[str, dict] = {}
        per_run: dict[str, list[CostBreakdown]] = {}
        for adv_name, advs in adversaries.items():
            for name in algos:
                key = name if adv_name == "none" else f"{name}@{adv_name}"
                costs: list[CostBreakdown] = []
                for i in range(cfg.n_episodes):
                    algo = get_execution(name)()
                    task = ExecutionTask(
                        side=cfg.side,
                        total_qty=qty,
                        horizon=horizon,
                        step=cfg.step,
                    )
                    adv_objs = tuple(a() for a in advs)
                    cb = run_episode(
                        algo, task, flow_cls, params, seed_kwargs,
                        sim_seed=cfg.seed + i,
                        tick_size=cfg.tick_size,
                        adversaries=adv_objs,
                    )
                    costs.append(cb)
                per_run[key] = costs
                table[key] = summarize(costs)
                if verbose:
                    s = table[key]
                    print(
                        f"{regime:>9} {key:<22} IS {s['mean_is_bps']:+.2f}±"
                        f"{s['std_is_bps']:.2f}bps CVaR {s['cvar95_is_bps']:+.2f} "
                        f"done {s['completion_rate']:.0%}"
                    )
        # paired comparison vs plain twap on identical seeds
        if "twap" in per_run:
            base = np.array([c.is_bps for c in per_run["twap"]])
            for key, costs in per_run.items():
                if key == "twap" or "@" in key:
                    continue
                d = np.array([c.is_bps for c in costs]) - base
                m, lo, hi = paired_bootstrap(d, cfg.n_bootstrap, cfg.seed)
                out["paired_vs_twap"][(regime, key)] = {
                    "mean_diff_bps": m,
                    "ci95": (lo, hi),
                }
        out["regimes"][regime] = table
    return out
