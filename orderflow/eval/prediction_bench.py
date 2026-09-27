"""Prediction benchmark: per-target NLL/accuracy/Brier/AUC on held-out sims.

Predictors are trained on one set of episodes and scored on a disjoint set,
per regime. NLL uses the same Gaussian-link convention as scripts/train_model
(mu, sigma) for the two regression targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import orderflow.models  # noqa: F401 - populates the predictor registry
from orderflow.book.book import LimitOrderBook
from orderflow.models.dataset import OrderFlowDataset, build_sequences
from orderflow.models.features import tokenize
from orderflow.models.metrics import accuracy, auc, brier, nll
from orderflow.models.targets import make_targets
from orderflow.models.transformer import BIN_TARGETS
from orderflow.registry import get_predictor, list_predictor
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


@dataclass
class PredBenchConfig:
    flow: str = "hawkes"
    n_train: int = 6
    n_test: int = 3
    horizon: float = 600.0
    seq_len: int = 128
    stride: int = 64
    h_events: int = 20
    seed: int = 0


def gen_episodes(regime: str, flow: str, n: int, horizon: float, seed: int):
    cls, params, skw = get_regime(regime, flow=flow)
    out = []
    for i in range(n):
        book = LimitOrderBook()
        rng = np.random.default_rng(seed * 1000 + i)
        seed_book(book, rng=rng, **skw)
        sim = Simulator(
            book, cls(dict(params)), seed=seed * 1000 + i,
            record_state_every_event=True,
        )
        out.append(sim.run(horizon))
    return out


def score_predictor(pred, ds: OrderFlowDataset) -> dict[str, float]:
    """Per-target metrics over all valid positions in the dataset."""
    res: dict[str, float] = {}
    probs: dict[str, list] = {}
    ys: dict[str, list] = {}
    for i in range(len(ds)):
        batch = {k: v[i].unsqueeze(0) for k, v in ds.tensors.items()}
        out = pred.predict(batch)
        valid = batch["valid"].reshape(-1).numpy()
        for k in pred.targets:
            if k not in out or k in ("next_level", "next_size", "next_log_dt"):
                # keep the headline targets; aux factorisation targets are NLL'd
                # by train_model already and add noise here
                continue
            y = batch[k].reshape(-1).numpy()[valid]
            if k in ("short_vol",):
                mu, sigma = out[k]
                mu = np.asarray(mu).reshape(-1)[valid]
                sigma = np.asarray(sigma).reshape(-1)[valid]
                ll = -0.5 * ((y - mu) / sigma) ** 2 - np.log(sigma)
                res[f"{k}_nll"] = res.get(f"{k}_nll", 0.0) + float(-np.mean(ll))
                res[f"_n_{k}"] = res.get(f"_n_{k}", 0) + 1
            elif k in BIN_TARGETS:
                p = np.asarray(out[k]).reshape(-1)[valid]
                probs.setdefault(k, []).append(p)
                ys.setdefault(k, []).append(y)
            else:
                p = np.asarray(out[k]).reshape(-1, out[k].shape[-1])[valid]
                probs.setdefault(k, []).append(p)
                ys.setdefault(k, []).append(y)
    for k in ("short_vol",):
        n = res.pop(f"_n_{k}", 1)
        res[f"{k}_nll"] = res.get(f"{k}_nll", 0.0) / max(n, 1)
    for k, ps in probs.items():
        p = np.concatenate(ps)
        y = np.concatenate(ys[k])
        if k in BIN_TARGETS:
            bp = np.column_stack([1 - p, p])
            res[f"{k}_nll"] = nll(bp, y)
            res[f"{k}_brier"] = brier(bp, y)
            res[f"{k}_auc"] = auc(p, y)
        else:
            res[f"{k}_nll"] = nll(p, y)
            res[f"{k}_acc"] = accuracy(p, y)
            res[f"{k}_brier"] = brier(p, y)
    return res


def _build_ds(results, size_bins, cfg: PredBenchConfig) -> OrderFlowDataset:
    streams = [tokenize(r, K=5, B=8, size_bins=size_bins) for r in results]
    tgts = [make_targets(s, r, h_events=cfg.h_events) for s, r in zip(streams, results)]
    return OrderFlowDataset(
        build_sequences(streams, tgts, seq_len=cfg.seq_len, stride=cfg.stride)
    )


def run_pred_bench(
    predictors: list[str] | None = None,
    regimes: list[str] = ("normal",),
    cfg: PredBenchConfig | None = None,
    verbose: bool = True,
) -> dict:
    """{regime: {predictor: {metric: value}}}."""
    cfg = cfg or PredBenchConfig()
    predictors = predictors or list_predictor()
    out: dict = {}
    for regime in regimes:
        results = gen_episodes(
            regime, cfg.flow, cfg.n_train + 1 + cfg.n_test, cfg.horizon, cfg.seed
        )
        train_res = results[: cfg.n_train]
        val_res = results[cfg.n_train : cfg.n_train + 1]
        test_res = results[cfg.n_train + 1 :]
        # shared size bins fitted on train episodes only (matches make_splits)
        train_streams = [tokenize(r) for r in train_res]
        size_bins = np.unique(
            np.concatenate([s.size_bins for s in train_streams])
        )
        train_ds = _build_ds(train_res, size_bins, cfg)
        val_ds = _build_ds(val_res, size_bins, cfg)
        test_ds = _build_ds(test_res, size_bins, cfg)
        table: dict[str, dict] = {}
        for name in predictors:
            cls = get_predictor(name)
            pred = cls()
            pred.fit(train_ds, val_ds)
            table[name] = score_predictor(pred, test_ds)
            if verbose:
                row = " ".join(
                    f"{k}={v:.3f}" for k, v in sorted(table[name].items())
                )
                print(f"{regime:>9} {name:<14} {row}")
        out[regime] = table
    return out
