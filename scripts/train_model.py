"""Generate sims, build datasets, train the Transformer + baselines, compare.

Usage: python scripts/train_model.py --flow hawkes --regime normal \
        --episodes 8 --horizon 600 --seq-len 128 --quick
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from orderflow.book.book import LimitOrderBook
from orderflow.models.baselines import LogisticBaseline, MarkovBaseline, MLPBaseline
from orderflow.models.dataset import make_splits
from orderflow.models.metrics import nll
from orderflow.models.train import TrainConfig, TransformerPredictor
from orderflow.models.transformer import TransformerConfig
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


def gen_episodes(flow, regime, n, horizon, seed):
    cls, params, skw = get_regime(regime, flow=flow)
    out = []
    for i in range(n):
        book = LimitOrderBook()
        rng = np.random.default_rng(seed * 1000 + i)
        seed_book(book, rng=rng, **skw)
        sim = Simulator(book, cls(params), seed=seed * 1000 + i, record_state_every_event=True)
        out.append(sim.run(horizon))
    return out


def evaluate_predictor(pred, ds):
    """Mean NLL over valid positions per target."""
    res: dict[str, list[float]] = {}
    for i in range(len(ds)):
        batch = {k: v[i].unsqueeze(0) for k, v in ds.tensors.items()}
        out = pred.predict(batch)
        valid = batch["valid"].reshape(-1).numpy()
        for k in pred.targets:
            if k not in out:
                continue
            y = batch[k].reshape(-1).numpy()[valid]
            if k in ("short_vol", "next_log_dt"):
                mu, sigma = out[k]
                mu = np.asarray(mu).reshape(-1)[valid]
                sigma = np.asarray(sigma).reshape(-1)[valid]
                ll = -0.5 * ((y - mu) / sigma) ** 2 - np.log(sigma)
                res.setdefault(k, []).append(float(-np.mean(ll)))
            elif k in ("cancel_prob", "queue_depletion"):
                p = np.asarray(out[k]).reshape(-1)[valid]
                bp = np.column_stack([1 - p, p])
                res.setdefault(k, []).append(nll(bp, y))
            else:
                p = np.asarray(out[k]).reshape(-1, out[k].shape[-1])[valid]
                res.setdefault(k, []).append(nll(p, y))
    return {k: float(np.mean(v)) for k, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="hawkes")
    ap.add_argument("--regime", default="normal")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--horizon", type=float, default=600.0)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    t0 = time.time()

    if args.quick:
        episodes, horizon = min(args.episodes, 6), min(args.horizon, 300.0)
        mcfg = TransformerConfig(d_model=64, n_layers=2, n_heads=4, seq_len=args.seq_len)
        tcfg = TrainConfig(epochs=2, batch=64)
    else:
        episodes, horizon = args.episodes, args.horizon
        mcfg = TransformerConfig(seq_len=args.seq_len)
        tcfg = TrainConfig()

    results = gen_episodes(args.flow, args.regime, episodes, horizon, args.seed)
    train_ds, val_ds, _test_ds = make_splits(
        results, seq_len=args.seq_len, stride=args.seq_len // 2
    )
    print(f"episodes={episodes} train_windows={len(train_ds)} val_windows={len(val_ds)}")

    predictors = [
        ("markov", MarkovBaseline()),
        ("logistic", LogisticBaseline()),
        ("mlp", MLPBaseline()),
        ("transformer", TransformerPredictor(mcfg, tcfg)),
    ]
    rows = {}
    for name, pred in predictors:
        ts = time.time()
        pred.fit(train_ds, val_ds)
        rows[name] = evaluate_predictor(pred, val_ds)
        print(f"{name:12s} fit+eval in {time.time() - ts:.1f}s")

    all_targets = sorted({k for v in rows.values() for k in v})
    lines = ["| target | " + " | ".join(rows) + " |", "|" + "---|" * (len(rows) + 1)]
    for tgt in all_targets:
        lines.append(
            "| "
            + tgt
            + " | "
            + " | ".join(f"{rows[m].get(tgt, float('nan')):.4f}" for m in rows)
            + " |"
        )
    table = "\n".join(lines)
    print("\nval NLL per target\n" + table)
    out = Path("artifacts/models")
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text("# Predictor comparison (val NLL)\n\n" + table + "\n")
    print(f"\nwall time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
