"""Baseline predictors: Markov, logistic, MLP — all under the Predictor protocol."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from orderflow.models.features import STATE_DIM
from orderflow.registry import register_predictor

N_TYPES = 6


def _collate_np(ds) -> dict[str, np.ndarray]:
    """Pull a whole dataset into numpy arrays (baselines train full-batch-ish)."""
    out: dict[str, list] = {k: [] for k in ds.tensors}
    for i in range(len(ds)):
        for k, v in ds.tensors.items():
            out[k].append(v[i].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def _flat(ds):
    """Flatten (batch, seq, ...) -> (N, ...) keeping only valid positions."""
    data = _collate_np(ds)  # already (N, ...) flattened over batch*seq
    valid = data.pop("valid").astype(bool)
    return {k: v[valid] for k, v in data.items()}, valid


@register_predictor("markov")
class MarkovBaseline:
    """Empirical 6x6 Laplace-smoothed transition matrix for next_type."""

    name = "markov"
    targets: ClassVar = ["next_type"]

    def __init__(self) -> None:
        self.trans = np.ones((N_TYPES, N_TYPES)) / N_TYPES

    def fit(self, train, val=None) -> MarkovBaseline:
        counts = np.ones((N_TYPES, N_TYPES))
        for i in range(len(train)):
            b = train[i]
            t = b["type_id"].numpy()
            v = b["valid"].numpy()
            src = t[:-1][v[:-1]]
            nxt = t[1:][v[:-1]]
            for a, b_ in zip(src, nxt):
                counts[a, b_] += 1
        self.trans = counts / counts.sum(axis=1, keepdims=True)
        return self

    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        t = batch["type_id"].numpy()
        return {"next_type": self.trans[t]}

    def predict_proba(self, type_ids: np.ndarray) -> np.ndarray:
        return self.trans[np.asarray(type_ids)]


class _SupervisedBase:
    """Shared fit/predict plumbing for torch models on flattened states."""

    def _targets_and_shapes(self):
        raise NotImplementedError

    def _fit_loop(self, model: nn.Module, ds, epochs: int = 60, lr: float = 1e-2):
        flat, _ = _flat(ds)
        X = torch.tensor(np.asarray(flat["state"]), dtype=torch.float32)
        Ys = {k: torch.tensor(np.asarray(flat[k])) for k in self.targets if k in flat}
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X)
            loss = self._loss(out, Ys)
            loss.backward()
            opt.step()
        return model

    def _loss(self, out, Ys):
        raise NotImplementedError


@register_predictor("logistic")
class LogisticBaseline(_SupervisedBase):
    """One Linear head per target: CE for categorical, BCE for binary,
    linear-Gaussian (mu + residual sigma) for short_vol / next_log_dt."""

    name = "logistic"
    targets: ClassVar = [
        "mid_move",
        "spread_change",
        "cancel_prob",
        "queue_depletion",
        "next_type",
        "short_vol",
        "next_log_dt",
    ]
    _NCLASS: ClassVar = {"mid_move": 3, "spread_change": 5, "next_type": N_TYPES}
    _GAUSS: ClassVar = {"short_vol", "next_log_dt"}

    def __init__(self, state_dim: int = STATE_DIM) -> None:
        self.heads = nn.ModuleDict(
            {k: nn.Linear(state_dim, self._NCLASS.get(k, 1)) for k in self.targets}
        )
        self.gauss_sigma: dict[str, float] = {}
        self._model = _LinearHeads(self.heads)

    def _loss(self, out, Ys):
        loss = 0.0
        for k in self.targets:
            if k not in Ys:
                continue
            if k in self._NCLASS:
                loss = loss + F.cross_entropy(out[k], Ys[k].long())
            elif k in self._GAUSS:
                y = Ys[k].float()
                m = out[k].squeeze(-1)
                loss = loss + F.mse_loss(m, y)
            else:
                loss = loss + F.binary_cross_entropy_with_logits(out[k].squeeze(-1), Ys[k].float())
        return loss

    def fit(self, train, val=None) -> LogisticBaseline:
        self._fit_loop(self._model, train)
        flat, _ = _flat(train)
        for k in self._GAUSS:
            if k in flat:
                X = torch.tensor(np.asarray(flat["state"]), dtype=torch.float32)
                with torch.no_grad():
                    resid = flat[k] - self.heads[k](X).squeeze(-1).numpy()
                self.gauss_sigma[k] = float(np.std(resid)) or 1.0
        return self

    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray | tuple]:
        X = batch["state"].float()
        B_, T, _ = X.shape
        flat_X = X.reshape(-1, X.shape[-1])
        out: dict[str, np.ndarray | tuple] = {}
        with torch.no_grad():
            for k in self.targets:
                h = self.heads[k](flat_X)
                if k in self._NCLASS:
                    out[k] = F.softmax(h, -1).reshape(B_, T, -1).numpy()
                elif k in self._GAUSS:
                    mu = h.squeeze(-1).reshape(B_, T).numpy()
                    out[k] = (mu, np.full_like(mu, self.gauss_sigma.get(k, 1.0)))
                else:
                    out[k] = torch.sigmoid(h).squeeze(-1).reshape(B_, T).numpy()
        return out


@register_predictor("mlp")
class MLPBaseline(_SupervisedBase):
    """2-layer MLP trunk with one head per target (same heads as logistic)."""

    name = "mlp"
    targets: ClassVar = LogisticBaseline.targets
    _NCLASS: ClassVar = LogisticBaseline._NCLASS
    _GAUSS: ClassVar = LogisticBaseline._GAUSS

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 64) -> None:
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.heads = nn.ModuleDict(
            {k: nn.Linear(hidden, self._NCLASS.get(k, 1)) for k in self.targets}
        )
        self.gauss_sigma: dict[str, float] = {}
        self._model = _MLPJoint(self.trunk, self.heads)

    def _loss(self, out, Ys):
        return LogisticBaseline._loss(self, out, Ys)

    def fit(self, train, val=None) -> MLPBaseline:
        self._fit_loop(self._model, train, epochs=80)
        flat, _ = _flat(train)
        X = torch.tensor(np.asarray(flat["state"]), dtype=torch.float32)
        with torch.no_grad():
            emb = self.trunk(X)
            for k in self._GAUSS:
                if k in flat:
                    resid = flat[k] - self.heads[k](emb).squeeze(-1).numpy()
                    self.gauss_sigma[k] = float(np.std(resid)) or 1.0
        return self

    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray | tuple]:
        X = batch["state"].float()
        B_, T, _ = X.shape
        flat_X = X.reshape(-1, X.shape[-1])
        with torch.no_grad():
            emb = self.trunk(flat_X)
            out: dict[str, np.ndarray | tuple] = {}
            for k in self.targets:
                h = self.heads[k](emb)
                if k in self._NCLASS:
                    out[k] = F.softmax(h, -1).reshape(B_, T, -1).numpy()
                elif k in self._GAUSS:
                    mu = h.squeeze(-1).reshape(B_, T).numpy()
                    out[k] = (mu, np.full_like(mu, self.gauss_sigma.get(k, 1.0)))
                else:
                    out[k] = torch.sigmoid(h).squeeze(-1).reshape(B_, T).numpy()
        return out


class _MLPJoint(nn.Module):
    def __init__(self, trunk, heads) -> None:
        super().__init__()
        self.trunk = trunk
        self.heads = heads

    def forward(self, x):
        emb = self.trunk(x)
        return {k: h(emb) for k, h in self.heads.items()}


class _LinearHeads(nn.Module):
    def __init__(self, heads) -> None:
        super().__init__()
        self.heads = heads

    def forward(self, x):
        return {k: h(x) for k, h in self.heads.items()}
