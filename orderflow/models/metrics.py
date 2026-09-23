"""Evaluation metrics shared by training and the eval harness (no sklearn)."""

from __future__ import annotations

import numpy as np


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.asarray(probs).argmax(axis=1) == np.asarray(y)))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier: mean squared error of probs vs one-hot targets."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    oh = np.zeros_like(probs)
    oh[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Binary AUC via the Mann–Whitney rank statistic."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.zeros(len(counts))
    np.add.at(csum, inv, ranks)
    avg = csum / counts
    r = avg[inv]
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
