"""OrderFlowTransformer: causal Transformer over tokenised order-flow events."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from orderflow.models.features import STATE_DIM

N_TYPES = 6
CAT_TARGETS = {"next_type": N_TYPES, "mid_move": 3, "spread_change": 5}
BIN_TARGETS = ("cancel_prob", "queue_depletion")


@dataclass
class TransformerConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    K: int = 5
    B: int = 8
    state_dim: int = STATE_DIM
    seq_len: int = 128
    dropout: float = 0.1
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "next_type": 1.0,
            "next_level": 0.5,
            "next_size": 0.5,
            "next_log_dt": 0.5,
            "mid_move": 1.0,
            "spread_change": 0.5,
            "cancel_prob": 0.5,
            "short_vol": 0.5,
            "queue_depletion": 0.5,
        }
    )


@dataclass
class NextEventDistribution:
    """Factorised distribution at the last position of a forward pass."""

    type_probs: np.ndarray  # (batch, 6)
    level_probs: np.ndarray  # (batch, K+1)
    size_probs: np.ndarray  # (batch, B)
    dt_mu: np.ndarray  # (batch,)
    dt_sigma: np.ndarray  # (batch,)

    def sample(self, rng: np.random.Generator, i: int = 0):
        """Sample (type_id, level_id, size_id, dt) for batch element i."""
        type_id = int(rng.choice(len(self.type_probs[i]), p=self.type_probs[i]))
        level_id = int(rng.choice(len(self.level_probs[i]), p=self.level_probs[i]))
        size_id = int(rng.choice(len(self.size_probs[i]), p=self.size_probs[i]))
        log_dt = rng.normal(self.dt_mu[i], self.dt_sigma[i])
        return type_id, level_id, size_id, float(np.exp(log_dt))


class OrderFlowTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TransformerConfig()
        c = self.cfg
        self.type_emb = nn.Embedding(N_TYPES, c.d_model)
        self.level_emb = nn.Embedding(c.K + 1, c.d_model)
        self.size_emb = nn.Embedding(c.B, c.d_model)
        self.dt_proj = nn.Linear(1, c.d_model)
        self.state_mlp = nn.Sequential(
            nn.Linear(c.state_dim, c.d_model), nn.GELU(), nn.Linear(c.d_model, c.d_model)
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, c.seq_len, c.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=4 * c.d_model,
            dropout=c.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=c.n_layers)
        self.head_next_type = nn.Linear(c.d_model, N_TYPES)
        self.head_next_level = nn.Linear(c.d_model, c.K + 1)
        self.head_next_size = nn.Linear(c.d_model, c.B)
        self.head_next_dt = nn.Linear(c.d_model, 2)  # (mu, log_sigma) of log dt
        self.head_mid_move = nn.Linear(c.d_model, 3)
        self.head_spread_change = nn.Linear(c.d_model, 5)
        self.head_cancel_prob = nn.Linear(c.d_model, 1)
        self.head_short_vol = nn.Linear(c.d_model, 1)
        self.head_queue_depl = nn.Linear(c.d_model, 1)
        self.short_vol_logvar = nn.Parameter(torch.zeros(1))  # learned scalar log-var

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _, T = batch["type_id"].shape
        x = (
            self.type_emb(batch["type_id"])
            + self.level_emb(batch["level_id"])
            + self.size_emb(batch["size_id"])
            + self.dt_proj(batch["log_dt"].unsqueeze(-1))
            + self.state_mlp(batch["state"])
            + self.pos_emb[:, :T]
        )
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.encoder(x, mask=mask)
        dt = self.head_next_dt(h)
        return {
            "next_type": self.head_next_type(h),
            "next_level": self.head_next_level(h),
            "next_size": self.head_next_size(h),
            "next_dt": dt,
            "mid_move": self.head_mid_move(h),
            "spread_change": self.head_spread_change(h),
            "cancel_prob": self.head_cancel_prob(h).squeeze(-1),
            "short_vol": self.head_short_vol(h).squeeze(-1),
            "queue_depletion": self.head_queue_depl(h).squeeze(-1),
        }

    def loss(
        self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        valid = batch["valid"]
        denom = valid.sum().clamp(min=1)
        per_head: dict[str, torch.Tensor] = {}
        for key in CAT_TARGETS:
            logits = out[key]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), batch[key].reshape(-1), reduction="none"
            ).reshape_as(valid)
            per_head[key] = (ce * valid).sum() / denom
        # next level/size use the same CE machinery
        for key, head_key in (("next_level", "next_level"), ("next_size", "next_size")):
            logits = out[head_key]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), batch[key].reshape(-1), reduction="none"
            ).reshape_as(valid)
            per_head[key] = (ce * valid).sum() / denom
        for key in BIN_TARGETS:
            bce = F.binary_cross_entropy_with_logits(out[key], batch[key].float(), reduction="none")
            per_head[key] = (bce * valid).sum() / denom
        # short_vol: Gaussian NLL with learned scalar log-variance
        y = batch["short_vol"]
        lv = self.short_vol_logvar
        nll = 0.5 * ((y - out["short_vol"]) ** 2 * torch.exp(-lv) + lv)
        per_head["short_vol"] = (nll * valid).sum() / denom
        # next_log_dt: log-normal NLL <=> Gaussian NLL on log dt (+ log|dt| const dropped)
        dt = out["next_dt"]
        mu, log_sigma = dt[..., 0], dt[..., 1].clamp(-6, 6)
        sigma2 = torch.exp(2 * log_sigma)
        nll_dt = 0.5 * (batch["next_log_dt"] - mu) ** 2 / sigma2 + log_sigma
        per_head["next_log_dt"] = (nll_dt * valid).sum() / denom
        total = sum(self.cfg.loss_weights.get(k, 1.0) * v for k, v in per_head.items())
        return total, {k: float(v.detach()) for k, v in per_head.items()}

    @torch.no_grad()
    def predict_next_event_distribution(
        self, batch: dict[str, torch.Tensor]
    ) -> NextEventDistribution:
        out = self.forward(batch)
        return NextEventDistribution(
            type_probs=F.softmax(out["next_type"][:, -1], dim=-1).cpu().numpy(),
            level_probs=F.softmax(out["next_level"][:, -1], dim=-1).cpu().numpy(),
            size_probs=F.softmax(out["next_size"][:, -1], dim=-1).cpu().numpy(),
            dt_mu=out["next_dt"][:, -1, 0].cpu().numpy(),
            dt_sigma=torch.exp(out["next_dt"][:, -1, 1].clamp(-6, 6)).cpu().numpy(),
        )
