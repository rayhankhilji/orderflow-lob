import numpy as np
import torch

from orderflow.book.book import LimitOrderBook
from orderflow.models.baselines import LogisticBaseline, MarkovBaseline, MLPBaseline
from orderflow.models.dataset import OrderFlowDataset, make_splits
from orderflow.models.features import STATE_DIM
from orderflow.models.metrics import nll
from orderflow.models.train import TrainConfig, train
from orderflow.models.transformer import OrderFlowTransformer, TransformerConfig
from orderflow.sim.regimes import get_regime
from orderflow.sim.simulator import Simulator, seed_book


def test_markov_row_stochastic():
    m = MarkovBaseline()
    counts_ds = OrderFlowDataset(
        {
            "type_id": torch.tensor([[0, 1, 0, 1, 0]]),
            "valid": torch.tensor([[True] * 5]),
            "level_id": torch.zeros(1, 5, dtype=torch.int64),
            "size_id": torch.zeros(1, 5, dtype=torch.int64),
            "log_dt": torch.zeros(1, 5),
            "state": torch.zeros(1, 5, STATE_DIM),
            "next_type": torch.tensor([[1, 0, 1, 0, 0]]),
        }
    )
    m.fit(counts_ds)
    assert np.allclose(m.trans.sum(axis=1), 1.0)
    assert m.trans[0, 1] > m.trans[0, 0]


def test_logistic_beats_uniform_on_separable():
    # synthetic: target = 1{state[:,0] > 0}
    n = 400
    X = np.random.default_rng(0).standard_normal((n, STATE_DIM))
    y = (X[:, 0] > 0).astype(int)
    tensors = {
        "type_id": torch.zeros(n, 4, dtype=torch.int64),
        "level_id": torch.zeros(n, 4, dtype=torch.int64),
        "size_id": torch.zeros(n, 4, dtype=torch.int64),
        "log_dt": torch.zeros(n, 4),
        "state": torch.tensor(X).unsqueeze(1).repeat(1, 4, 1).float(),
        "mid_move": torch.tensor(y).unsqueeze(1).repeat(1, 4),
        "spread_change": torch.zeros(n, 4, dtype=torch.int64),
        "next_type": torch.zeros(n, 4, dtype=torch.int64),
        "next_level": torch.zeros(n, 4, dtype=torch.int64),
        "next_size": torch.zeros(n, 4, dtype=torch.int64),
        "next_log_dt": torch.zeros(n, 4),
        "cancel_prob": torch.zeros(n, 4, dtype=torch.int64),
        "queue_depletion": torch.zeros(n, 4, dtype=torch.int64),
        "short_vol": torch.zeros(n, 4),
        "valid": torch.ones(n, 4, dtype=torch.bool),
    }
    ds = OrderFlowDataset(tensors)
    log = LogisticBaseline().fit(ds)
    out = log.predict({k: v[:16] for k, v in tensors.items()})
    p = out["mid_move"].reshape(-1, 3)
    yy = tensors["mid_move"][:16].reshape(-1).numpy()
    uniform = np.full_like(p, 1 / 3)
    assert nll(p, yy) < nll(uniform, yy)


def _two_short_sims():
    res = []
    for i in range(2):
        book = LimitOrderBook()
        cls, params, skw = get_regime("normal", "zi")
        seed_book(book, rng=np.random.default_rng(i), **skw)
        sim = Simulator(book, cls(params), seed=i, record_state_every_event=True)
        res.append(sim.run(60))
    return res


def test_train_smoke():
    torch.manual_seed(0)
    train_ds, val_ds, _ = make_splits(_two_short_sims() * 3, seq_len=64, stride=32)
    assert len(train_ds) > 0
    model = OrderFlowTransformer(TransformerConfig(d_model=32, n_layers=1, n_heads=2, seq_len=64))
    hist = train(
        model,
        train_ds,
        val_ds,
        TrainConfig(epochs=1, batch=16, name="smoke"),
        out_dir="/tmp/of_models",
    )
    hist["train_loss"][0]
    assert len(hist["train_loss"]) == 1
    # compare first vs last batch loss is unavailable post-hoc; check finite val loss
    assert np.isfinite(hist["val_loss"][0])
    mlp = MLPBaseline().fit(train_ds)
    out = mlp.predict({k: v[:2] for k, v in train_ds.tensors.items()})
    assert "mid_move" in out
