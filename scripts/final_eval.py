"""Final evaluation: train learned+RL checkpoints, full bench, leaderboard.

Usage: python scripts/final_eval.py [--skip-train]

1. Generates episodes on `normal`, trains MLPBaseline (state -> targets),
   saves artifacts/models/mlp_mid_move.pt for the `learned` policy.
2. Trains PPO on the scaled exec task (20k/600s — obs and actions are
   normalized, so the policy transfers to the 100k/1800s flagship),
   saves artifacts/models/ppo_exec.pt.
3. Runs the full matrix: all registered algos x {exec, normal, volatile,
   thin, stressed} clean + {exec, thin} under the combined five adversaries.
4. Writes LEADERBOARD.md + survivors.json + artifacts/final_bench.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import orderflow.execution  # noqa: F401
from orderflow.book.book import LimitOrderBook
from orderflow.book.types import Side
from orderflow.eval.adversarial import ADVERSARIES
from orderflow.eval.execution_bench import run_episode
from orderflow.eval.select import write_leaderboard
from orderflow.execution.cost import summarize
from orderflow.execution.learned import LearnedPolicy, TorchSignal
from orderflow.execution.rl.env import N_ACTIONS, OBS_DIM, make_exec_env
from orderflow.execution.rl.policy import RLExecution
from orderflow.execution.rl.ppo import PPO
from orderflow.execution.types import ExecutionTask
from orderflow.models.baselines import MLPBaseline
from orderflow.models.dataset import make_splits
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book

ART = Path("artifacts")
MODELS = ART / "models"

ALGO_NAMES = [
    "twap", "twap_passive", "twap_capped", "vwap",
    "ac", "ac_adaptive", "learned", "rl",
]
TASKS = {
    "exec": (1800.0, 100_000, 2),      # horizon, qty, n_episodes
    "normal": (600.0, 4_000, 4),
    "volatile": (600.0, 3_000, 4),
    "thin": (600.0, 800, 4),
    "stressed": (600.0, 1_500, 4),
}
ADV_TASKS = {"exec": (1800.0, 100_000, 2), "thin": (600.0, 800, 4)}
COMBINED = ("spoofer", "igniter", "withdrawer", "frontrunner", "pennyjumper")


def gen_episodes(n: int, horizon: float, seed: int):
    cls, params, skw = get_regime("normal", flow="hawkes")
    out = []
    for i in range(n):
        book = LimitOrderBook()
        rng = np.random.default_rng(seed * 1000 + i)
        seed_book(book, rng=rng, **skw)
        sim = Simulator(
            book, cls(params), seed=seed * 1000 + i, record_state_every_event=True
        )
        out.append(sim.run(horizon))
    return out


def train_mlp(seed: int) -> MLPBaseline:
    print("[final] generating episodes + fitting MLP baseline ...", flush=True)
    eps = gen_episodes(5, 600.0, seed)
    train_ds, val_ds, _ = make_splits(eps, seq_len=96, stride=48)
    mlp = MLPBaseline()
    mlp.fit(train_ds, val_ds)
    MODELS.mkdir(parents=True, exist_ok=True)
    mlp.save(MODELS / "mlp_mid_move.pt")
    print(f"[final] mlp saved ({len(train_ds)} train windows)", flush=True)
    return mlp


def load_mlp(path: Path) -> MLPBaseline:
    return MLPBaseline.load(path)


def train_ppo(seed: int, iters: int = 24, steps_per_iter: int = 480):
    print(f"[final] training PPO on scaled exec task ({iters}x{steps_per_iter}) ...", flush=True)
    env = make_exec_env(regime="exec", seed=seed, total_qty=20_000, horizon=600.0, step=10.0)
    ppo = PPO(OBS_DIM, N_ACTIONS, seed=seed)
    for it in range(iters):
        buf, stats = ppo.collect(env, steps_per_iter)
        losses = ppo.update(buf, last_v=stats["last_value"])
        print(
            f"  it{it}: eps={stats['episodes']} ret={stats['mean_ep_ret']:.3f} "
            f"H={losses['entropy']:.3f}",
            flush=True,
        )
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(ppo.net.state_dict(), MODELS / "ppo_exec.pt")
    return ppo.net


def make_algo(name: str, mlp, net):
    if name == "learned":
        if mlp is None:
            mlp = load_mlp(MODELS / "mlp_mid_move.pt")
        return LearnedPolicy(predictor=TorchSignal(mlp, Side.SELL))
    if name == "rl":
        return RLExecution().load_policy(str(MODELS / "ppo_exec.pt"))
    from orderflow.registry import get_execution

    return get_execution(name)()


def run_matrix(mlp, net, seed: int) -> dict:
    regimes: dict[str, dict] = {}
    for regime, (horizon, qty, eps) in TASKS.items():
        flow_cls, params, seed_kwargs = get_regime(regime, flow="hawkes")
        table = {}
        for name in ALGO_NAMES:
            t0 = time.time()
            costs = []
            for i in range(eps):
                costs.append(
                    run_episode(
                        make_algo(name, mlp, net),
                        ExecutionTask(side=Side.SELL, total_qty=qty, horizon=horizon, step=10.0),
                        flow_cls, params, seed_kwargs,
                        sim_seed=seed + i,
                    )
                )
            table[name] = summarize(costs)
            print(
                f"  {regime:>9} {name:14s} IS {table[name]['mean_is_bps']:+.2f} "
                f"done {table[name]['completion_rate']:.0%} ({time.time()-t0:.0f}s)",
                flush=True,
            )
        regimes[regime] = table

    for regime, (horizon, qty, eps) in ADV_TASKS.items():
        flow_cls, params, seed_kwargs = get_regime(regime, flow="hawkes")
        table = regimes.setdefault(regime, {})
        for name in ALGO_NAMES:
            costs = []
            for i in range(eps):
                adv = tuple(ADVERSARIES[k]() for k in COMBINED)
                costs.append(
                    run_episode(
                        make_algo(name, mlp, net),
                        ExecutionTask(side=Side.SELL, total_qty=qty, horizon=horizon, step=10.0),
                        flow_cls, params, seed_kwargs,
                        sim_seed=seed + 500 + i,
                        adversaries=adv,
                    )
                )
            table[f"{name}@combined"] = summarize(costs)
            print(
                f"  {regime:>9} {name}@combined IS {table[f'{name}@combined']['mean_is_bps']:+.2f} "
                f"done {table[f'{name}@combined']['completion_rate']:.0%}",
                flush=True,
            )
    return {"regimes": regimes, "paired_vs_twap": {}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--rl-iters", type=int, default=24)
    args = ap.parse_args()

    mlp = net = None
    if not args.skip_train:
        mlp = train_mlp(args.seed)
        net = train_ppo(args.seed, iters=args.rl_iters)

    print("[final] full matrix ...", flush=True)
    bench = run_matrix(mlp, net, args.seed)
    ART.mkdir(exist_ok=True)
    (ART / "final_bench.json").write_text(json.dumps(bench, indent=2))
    pred_path = Path("research/benchmark/pred_results.json")
    pred = json.loads(pred_path.read_text()) if pred_path.exists() else {}
    survivors = write_leaderboard(bench, pred, out_dir=".", md_path="LEADERBOARD.md")
    print(json.dumps(survivors["execution"], indent=2))
    print("[final] wrote LEADERBOARD.md + survivors.json + artifacts/final_bench.json")


if __name__ == "__main__":
    main()
