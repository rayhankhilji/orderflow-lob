"""Windowed sequence dataset over tokenised simulation episodes."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from orderflow.models.features import TokenStream, tokenize
from orderflow.models.targets import make_targets
from orderflow.sim.simulator import SimResult

TOKEN_FIELDS = ("type_id", "level_id", "size_id", "log_dt")
TARGET_KEYS = (
    "next_type",
    "next_level",
    "next_size",
    "next_log_dt",
    "mid_move",
    "spread_change",
    "cancel_prob",
    "short_vol",
    "queue_depletion",
)


def build_sequences(
    streams: list[TokenStream],
    targets: list[dict[str, np.ndarray]],
    seq_len: int = 128,
    stride: int = 64,
) -> dict[str, torch.Tensor]:
    """Slide windows over each episode's tokens; returns stacked tensors."""
    out: dict[str, list[np.ndarray]] = {
        k: [] for k in (*TOKEN_FIELDS, "state", *TARGET_KEYS, "valid")
    }
    for stream, tgt in zip(streams, targets):
        n = len(stream)
        for start in range(0, max(n - seq_len + 1, 0), stride):
            end = start + seq_len
            out["type_id"].append(stream.type_id[start:end])
            out["level_id"].append(stream.level_id[start:end])
            out["size_id"].append(stream.size_id[start:end])
            out["log_dt"].append(stream.log_dt[start:end])
            out["state"].append(stream.state[start:end])
            for k in TARGET_KEYS:
                out[k].append(np.asarray(tgt[k])[start:end])
            out["valid"].append(np.asarray(tgt["valid"])[start:end])
    tensors: dict[str, torch.Tensor] = {}
    for k, v in out.items():
        arr = np.asarray(v)
        if k in ("log_dt", "state", "short_vol", "next_log_dt"):
            tensors[k] = torch.tensor(arr, dtype=torch.float32)
        elif k == "valid":
            tensors[k] = torch.tensor(arr.astype(bool))
        else:
            tensors[k] = torch.tensor(arr, dtype=torch.int64)
    return tensors


class OrderFlowDataset(Dataset):
    """TensorDataset-like wrapper yielding dict batches."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors
        self._n = len(next(iter(tensors.values()))) if tensors else 0

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.tensors.items()}


def make_splits(
    results: list[SimResult],
    seq_len: int = 128,
    stride: int = 64,
    h_events: int = 20,
    K: int = 5,
    B: int = 8,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[OrderFlowDataset, OrderFlowDataset, OrderFlowDataset]:
    """Tokenise episodes, fit shared size bins on train, split 70/15/15."""
    n = len(results)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    parts = {
        "train": results[:n_train],
        "val": results[n_train : n_train + n_val],
        "test": results[n_train + n_val :],
    }
    # fit size bins on train episodes only, reuse everywhere
    train_streams = [tokenize(r, K=K, B=B) for r in parts["train"]]
    qty_bins = np.unique(
        np.concatenate([s.size_bins for s in train_streams]) if train_streams else np.empty(0)
    )
    datasets: dict[str, OrderFlowDataset] = {}
    for name, rs in parts.items():
        if not rs:
            datasets[name] = OrderFlowDataset({})
            continue
        streams = (
            train_streams
            if name == "train"
            else [tokenize(r, K=K, B=B, size_bins=qty_bins) for r in rs]
        )
        tgts = [make_targets(s, r, h_events=h_events) for s, r in zip(streams, rs)]
        datasets[name] = OrderFlowDataset(
            build_sequences(streams, tgts, seq_len=seq_len, stride=stride)
        )
    return datasets["train"], datasets["val"], datasets["test"]
