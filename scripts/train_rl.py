"""Train the PPO execution agent on the exec regime.

Usage:
    python scripts/train_rl.py --iters 30 --steps-per-iter 4000 \
        --out artifacts/models/ppo_exec.pt
For a smoke run: --iters 2 --steps-per-iter 400 --horizon 300 --qty 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from orderflow.execution.rl.env import N_ACTIONS, OBS_DIM, make_exec_env
from orderflow.execution.rl.ppo import PPO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="exec")
    ap.add_argument("--flow", default="hawkes")
    ap.add_argument("--qty", type=int, default=100_000)
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--risk-penalty", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--steps-per-iter", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/models/ppo_exec.pt")
    args = ap.parse_args()

    env = make_exec_env(
        regime=args.regime,
        flow=args.flow,
        seed=args.seed,
        total_qty=args.qty,
        horizon=args.horizon,
        step=args.step,
        risk_penalty=args.risk_penalty,
    )
    ppo = PPO(OBS_DIM, N_ACTIONS, seed=args.seed)
    log = []
    for it in range(args.iters):
        buf, stats = ppo.collect(env, args.steps_per_iter)
        losses = ppo.update(buf, last_v=stats["last_value"])
        log.append({**stats, **losses})
        print(
            f"iter {it}: eps={stats['episodes']} ret={stats['mean_ep_ret']:.3f} "
            f"pi={losses['pi_loss']:.4f} v={losses['v_loss']:.4f} "
            f"H={losses['entropy']:.3f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ppo.net.state_dict(), out)
    (out.parent / "ppo_exec_log.json").write_text(json.dumps(log, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
