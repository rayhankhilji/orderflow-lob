"""RL-research study — reward-shaping ablation + benchmark comparison.

Ablates the inventory-risk penalty in the execution reward
    r_t = -dIS_bps - risk_penalty * remaining_frac^2 * step/horizon
across {0.0, 0.5, 2.0} on a scaled task (20k shares / 600 s / 10 s grid),
then evaluates each trained policy deterministically via the RLExecution
wrapper on held-out seeds against TWAP and AC, all under common random
numbers. Writes results.json + a train-log json per setting.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

import orderflow.execution  # noqa: F401
from orderflow.book.types import Side
from orderflow.eval.execution_bench import run_episode
from orderflow.execution.cost import summarize
from orderflow.execution.rl.env import N_ACTIONS, OBS_DIM, make_exec_env
from orderflow.execution.rl.policy import RLExecution
from orderflow.execution.rl.ppo import PPO, ActorCritic
from orderflow.execution.types import ExecutionTask
from orderflow.registry import get_execution
from orderflow.sim.regimes import get_regime

OUT = Path(__file__).parent / "results.json"
QTY, HORIZON, STEP = 20_000, 600.0, 10.0
PENALTIES = [0.0, 0.5, 2.0]
TRAIN_ITERS = 8
STEPS_PER_ITER = 300
EVAL_SEEDS = [41, 42, 43, 44, 45]


def train_setting(risk_penalty: float, seed: int) -> tuple[ActorCritic, list]:
    env = make_exec_env(
        regime="exec",
        seed=seed,
        total_qty=QTY,
        horizon=HORIZON,
        step=STEP,
        risk_penalty=risk_penalty,
    )
    ppo = PPO(OBS_DIM, N_ACTIONS, seed=seed)
    log = []
    for it in range(TRAIN_ITERS):
        buf, stats = ppo.collect(env, STEPS_PER_ITER)
        losses = ppo.update(buf, last_v=stats["last_value"])
        log.append({**stats, **losses})
        print(
            f"    rp={risk_penalty} it{it}: eps={stats['episodes']} "
            f"ret={stats['mean_ep_ret']:.3f} H={losses['entropy']:.3f}",
            flush=True,
        )
    return ppo.net, log


def evaluate(policy: ActorCritic | None, seeds: list[int]) -> dict:
    flow_cls, params, seed_kwargs = get_regime("exec", flow="hawkes")
    task = ExecutionTask(side=Side.SELL, total_qty=QTY, horizon=HORIZON, step=STEP)
    costs = [
        run_episode(
            RLExecution(policy=policy),
            task,
            flow_cls,
            params,
            seed_kwargs,
            sim_seed=s,
        )
        for s in seeds
    ]
    return summarize(costs)


def evaluate_named(name: str, seeds: list[int]) -> dict:
    flow_cls, params, seed_kwargs = get_regime("exec", flow="hawkes")
    task = ExecutionTask(side=Side.SELL, total_qty=QTY, horizon=HORIZON, step=STEP)
    costs = [
        run_episode(
            get_execution(name)(), task, flow_cls, params, seed_kwargs, sim_seed=s
        )
        for s in seeds
    ]
    return summarize(costs)


def main() -> None:
    torch.manual_seed(0)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"[rl-study] task {QTY}/{HORIZON}s step={STEP} penalties={PENALTIES}")

    results: dict = {"seed": seed, "task": [QTY, HORIZON, STEP], "arms": {}}
    for rp in PENALTIES:
        t0 = time.time()
        print(f"[rl-study] training risk_penalty={rp} ...", flush=True)
        net, log = train_setting(rp, seed)
        evals = evaluate(net, EVAL_SEEDS)
        evals["train_seconds"] = round(time.time() - t0, 1)
        evals["final_ep_ret"] = log[-1]["mean_ep_ret"]
        evals["train_episodes"] = sum(x["episodes"] for x in log)
        results["arms"][f"ppo_rp{rp}"] = evals
        Path(__file__).parent.joinpath(f"train_log_rp{rp}.json").write_text(
            json.dumps(log, indent=2)
        )
        print(
            f"    -> eval mean IS {evals['mean_is_bps']:+.2f} bps "
            f"fill {evals['mean_fill_rate']:.1%} "
            f"(trained {evals['train_episodes']} eps)",
            flush=True,
        )

    print("[rl-study] baselines ...", flush=True)
    for name in ("twap", "ac"):
        results["arms"][name] = evaluate_named(name, EVAL_SEEDS)
        s = results["arms"][name]
        print(f"    {name}: mean IS {s['mean_is_bps']:+.2f} bps")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"[rl-study] wrote {OUT}")


if __name__ == "__main__":
    main()
