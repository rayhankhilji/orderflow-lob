"""Predictor protocol shared by baselines and the Transformer wrapper."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class Predictor(Protocol):
    name: str
    targets: list[str]

    def fit(self, train, val=None) -> Predictor: ...

    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray | tuple]:
        """Return per-target predictions: class probs (n, C) or (mu, sigma)."""
        ...
