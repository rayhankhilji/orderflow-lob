import numpy as np
import torch

from orderflow.models.transformer import OrderFlowTransformer, TransformerConfig

SEQ = 16


def make_batch(B=2, T=SEQ, cfg=None):
    cfg = cfg or TransformerConfig()
    g = torch.Generator().manual_seed(0)
    return {
        "type_id": torch.randint(0, 6, (B, T), generator=g),
        "level_id": torch.randint(0, cfg.K + 1, (B, T), generator=g),
        "size_id": torch.randint(0, cfg.B, (B, T), generator=g),
        "log_dt": torch.randn(B, T, generator=g),
        "state": torch.randn(B, T, cfg.state_dim, generator=g),
        "next_type": torch.randint(0, 6, (B, T), generator=g),
        "next_level": torch.randint(0, cfg.K + 1, (B, T), generator=g),
        "next_size": torch.randint(0, cfg.B, (B, T), generator=g),
        "next_log_dt": torch.randn(B, T, generator=g),
        "mid_move": torch.randint(0, 3, (B, T), generator=g),
        "spread_change": torch.randint(0, 5, (B, T), generator=g),
        "cancel_prob": torch.randint(0, 2, (B, T), generator=g),
        "short_vol": torch.randn(B, T, generator=g),
        "queue_depletion": torch.randint(0, 2, (B, T), generator=g),
        "valid": torch.ones(B, T, dtype=torch.bool),
    }


def test_forward_shapes():
    cfg = TransformerConfig(d_model=32, n_layers=1, n_heads=2, seq_len=SEQ)
    m = OrderFlowTransformer(cfg)
    out = m(make_batch(cfg=cfg))
    assert out["next_type"].shape == (2, SEQ, 6)
    assert out["next_level"].shape == (2, SEQ, cfg.K + 1)
    assert out["next_dt"].shape == (2, SEQ, 2)
    assert out["short_vol"].shape == (2, SEQ)


def test_loss_finite():
    cfg = TransformerConfig(d_model=32, n_layers=1, n_heads=2, seq_len=SEQ)
    m = OrderFlowTransformer(cfg)
    batch = make_batch(cfg=cfg)
    total, per = m.loss(m(batch), batch)
    assert torch.isfinite(total)
    assert "mid_move" in per and "next_log_dt" in per


def test_causal_mask():
    cfg = TransformerConfig(d_model=32, n_layers=2, n_heads=2, seq_len=SEQ)
    m = OrderFlowTransformer(cfg).eval()
    batch = make_batch(cfg=cfg)
    batch2 = {k: v.clone() for k, v in batch.items()}
    j = 5
    batch2["type_id"][:, j] = (batch2["type_id"][:, j] + 1) % 6
    batch2["state"][:, j] += 10.0
    with torch.no_grad():
        o1, o2 = m(batch), m(batch2)
    assert torch.allclose(o1["next_type"][:, :j], o2["next_type"][:, :j], atol=1e-5)
    assert not torch.allclose(o1["next_type"][:, j], o2["next_type"][:, j])


def test_predict_next_event_distribution_sample():
    cfg = TransformerConfig(d_model=32, n_layers=1, n_heads=2, seq_len=SEQ)
    m = OrderFlowTransformer(cfg)
    dist = m.predict_next_event_distribution(make_batch(cfg=cfg))
    rng = np.random.default_rng(0)
    t, l, s, dt = dist.sample(rng)
    assert 0 <= t < 6 and 0 <= l <= cfg.K and 0 <= s < cfg.B and dt > 0
