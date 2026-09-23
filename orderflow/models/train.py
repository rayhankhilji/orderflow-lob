"""Training loop + predictor wrapper for OrderFlowTransformer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from orderflow.models.metrics import accuracy, auc, brier, nll
from orderflow.models.transformer import (
    BIN_TARGETS,
    N_TYPES,
    OrderFlowTransformer,
    TransformerConfig,
)
from orderflow.registry import register_predictor


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 3e-4
    batch: int = 64
    weight_decay: float = 0.01
    patience: int = 3
    name: str = "transformer"


def _collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def evaluate(model: OrderFlowTransformer, ds, batch: int = 64) -> dict[str, float]:
    """Per-target val NLL/accuracy/Brier/AUC over valid positions."""
    model.eval()
    agg: dict[str, list] = {}
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=batch, collate_fn=_collate):
            out = model(b)
            valid = b["valid"].reshape(-1).numpy()
            for key in ("next_type", "mid_move", "spread_change", "next_level", "next_size"):
                p = F.softmax(out[key], -1).reshape(-1, out[key].shape[-1]).numpy()[valid]
                y = b[key].reshape(-1).numpy()[valid]
                agg.setdefault(key, []).append((p, y))
            for key in BIN_TARGETS:
                p = torch.sigmoid(out[key]).reshape(-1).numpy()[valid]
                y = b[key].reshape(-1).numpy()[valid]
                agg.setdefault(key, []).append((p, y))
            mu = out["short_vol"].reshape(-1).numpy()[valid]
            y = b["short_vol"].reshape(-1).numpy()[valid]
            agg.setdefault("short_vol", []).append((mu, y))
    metrics: dict[str, float] = {}
    for key, pairs in agg.items():
        p = np.concatenate([a for a, _ in pairs])
        y = np.concatenate([b for _, b in pairs])
        if key in BIN_TARGETS:
            binary_p = np.column_stack([1 - p, p])
            metrics[f"{key}_nll"] = nll(binary_p, y)
            metrics[f"{key}_brier"] = brier(binary_p, y)
            metrics[f"{key}_auc"] = auc(p, y)
        elif key == "short_vol":
            metrics[f"{key}_mse"] = float(np.mean((p - y) ** 2))
        else:
            metrics[f"{key}_nll"] = nll(p, y)
            metrics[f"{key}_acc"] = accuracy(p, y)
            metrics[f"{key}_brier"] = brier(p, y)
    return metrics


def train(
    model: OrderFlowTransformer,
    train_ds,
    val_ds,
    cfg: TrainConfig,
    out_dir: str | Path = "artifacts/models",
) -> dict:
    """AdamW + cosine schedule + early stopping on val total loss."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_ds) // cfg.batch + 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch)
    loader = DataLoader(
        train_ds, batch_size=cfg.batch, shuffle=True, collate_fn=_collate, drop_last=False
    )
    best_val = np.inf
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    for _epoch in range(cfg.epochs):
        losses = []
        for batch in loader:
            opt.zero_grad()
            out = model(batch)
            loss, _ = model.loss(out, batch)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        history["train_loss"].append(float(np.mean(losses)))
        # val loss
        model.eval()
        vlosses = []
        with torch.no_grad():
            for batch in DataLoader(val_ds, batch_size=cfg.batch, collate_fn=_collate):
                out = model(batch)
                loss, _ = model.loss(out, batch)
                vlosses.append(float(loss))
        model.train()
        vl = float(np.mean(vlosses)) if vlosses else np.inf
        history["val_loss"].append(vl)
        if vl < best_val - 1e-4:
            best_val = vl
            bad_epochs = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate(model, val_ds) if len(val_ds) else {}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": model.cfg.__dict__, "state_dict": model.state_dict()},
        out_path / f"{cfg.name}.pt",
    )
    with open(out_path / f"{cfg.name}_metrics.json", "w") as f:
        json.dump({"history": history, "val": metrics}, f, indent=2)
    history["val_metrics"] = metrics
    return history


@register_predictor("transformer")
class TransformerPredictor:
    """Wraps OrderFlowTransformer in the Predictor protocol."""

    name = "transformer"
    targets: ClassVar = [
        "next_type",
        "next_level",
        "next_size",
        "next_log_dt",
        "mid_move",
        "spread_change",
        "cancel_prob",
        "short_vol",
        "queue_depletion",
    ]

    def __init__(self, cfg: TransformerConfig | None = None, train_cfg: TrainConfig | None = None):
        self.model = OrderFlowTransformer(cfg)
        self.train_cfg = train_cfg or TrainConfig()

    def fit(self, train_ds, val_ds=None) -> TransformerPredictor:
        train(self.model, train_ds, val_ds or train_ds, self.train_cfg)
        return self

    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray | tuple]:
        self.model.eval()
        with torch.no_grad():
            out = self.model(batch)
        res: dict[str, np.ndarray | tuple] = {}
        for key in ("next_type", "next_level", "next_size", "mid_move", "spread_change"):
            res[key] = F.softmax(out[key], -1).numpy()
        for key in BIN_TARGETS:
            res[key] = torch.sigmoid(out[key]).numpy()
        res["short_vol"] = (
            out["short_vol"].numpy(),
            np.exp(0.5 * float(self.model.short_vol_logvar.detach()))
            * np.ones_like(out["short_vol"].numpy()),
        )
        dt = out["next_dt"].numpy()
        res["next_log_dt"] = (dt[..., 0], np.exp(np.clip(dt[..., 1], -6, 6)))
        return res


__all__ = ["N_TYPES", "TrainConfig", "TransformerPredictor", "evaluate", "train"]
