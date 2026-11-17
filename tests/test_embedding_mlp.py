"""Tests for src/embedding_mlp.py. They use small synthetic tables, so no data pack is needed."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding_mlp import (  # noqa: E402
    UNKNOWN_ID,
    EmbeddingMLP,
    TabularEncoder,
    TrainConfig,
    embedding_dims,
    load_model,
    predict_proba,
    sample_config,
    save_model,
    train_model,
)
from sklearn.metrics import average_precision_score  # noqa: E402

CATS = ["tier", "region"]


def make_table(n, seed=0):
    """Synthetic: buying depends on one numeric column and on the region (made-up data)."""
    rng = np.random.default_rng(seed)
    region = rng.choice([f"R{i:02d}" for i in range(20)], size=n)
    region_effect = {f"R{i:02d}": (2.0 if i < 5 else -1.0) for i in range(20)}
    visits = rng.normal(size=n)
    logit = -2.0 + 1.5 * visits + pd.Series(region).map(region_effect).to_numpy()
    X = pd.DataFrame(
        {
            "visits": visits,
            "avg_csat": np.where(rng.random(n) < 0.2, np.nan, rng.normal(4, 0.5, n)),
            "tier": rng.choice(["Bronze", "Silver", "Gold"], size=n),
            "region": region,
        }
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_encoder_learns_categories_from_train_only():
    X, _ = make_table(500)
    enc = TabularEncoder(CATS).fit(X)
    assert enc.numeric_cols_ == ["visits", "avg_csat"]
    assert enc.cardinalities == [3, 20]

    new = X.head(3).copy()
    new.loc[new.index[0], "region"] = "NEVER_SEEN"
    x_num, x_cat = enc.transform(new)
    assert x_num.dtype == np.float32 and x_cat.dtype == np.int64
    assert x_cat.shape == (3, 2)
    assert x_cat[0, 1] == UNKNOWN_ID  # unseen category -> 0
    assert (x_cat[1:] >= 1).all()  # seen categories -> 1..n
    assert not np.isnan(x_num).any()  # avg_csat gaps filled with the train median


def test_encoder_uses_train_statistics():
    X, _ = make_table(1000)
    train = X.iloc[:600]
    other = X.iloc[600:].assign(visits=X["visits"].iloc[600:] + 100)  # very different scale
    enc = TabularEncoder(CATS).fit(train)
    x_num, _ = enc.transform(other)
    # Scaled with the TRAIN mean, so the shifted column is far from 0, not re-centred.
    assert x_num[:, 0].mean() > 50


def test_encoder_logs_only_long_tailed_non_negative_columns():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "spend": rng.lognormal(3, 1.5, 2000),  # long right tail -> logged
            "visits": rng.normal(size=2000),  # symmetric, has negatives -> left alone
            "flag": (rng.random(2000) < 0.02).astype(int),  # rare 0/1 flag -> left alone
            "tier": "Gold",
        }
    )
    enc = TabularEncoder(["tier"]).fit(X)
    assert enc.log_cols_ == ["spend"]
    assert TabularEncoder(["tier"], log_skewed=False).fit(X).log_cols_ == []
    x_num, _ = enc.transform(X)
    assert abs(pd.Series(x_num[:, 0]).skew()) < 1  # the tail is gone after log1p

    # A negative value never seen in train is clipped to 0, not turned into NaN.
    odd = X.head(1).assign(spend=-5.0)
    assert not np.isnan(enc.transform(odd)[0]).any()


def test_embedding_dims_rule():
    assert embedding_dims([3, 6, 100]) == [3, 4, 16]  # 100 levels is capped at 16
    assert embedding_dims([100], max_dim=32) == [21]
    assert embedding_dims([2], scale=0.1) == [1]  # never below 1


def test_unknown_embedding_is_zero_and_frozen():
    model = EmbeddingMLP(n_numeric=2, cardinalities=[3, 20], emb_dims=[2, 4])
    for table in model.embeddings:
        assert torch.all(table.weight[UNKNOWN_ID] == 0)


def test_forward_shapes_and_probabilities():
    model = EmbeddingMLP(n_numeric=2, cardinalities=[3, 20], emb_dims=[2, 4], hidden=(8, 4))
    x_num = torch.randn(16, 2)
    x_cat = torch.randint(0, 4, (16, 2))
    assert model(x_num, x_cat).shape == (16,)
    p = predict_proba(model, (x_num.numpy(), x_cat.numpy()))
    assert p.shape == (16,) and ((p >= 0) & (p <= 1)).all()


def test_plain_mlp_when_no_text_columns():
    model = EmbeddingMLP(n_numeric=5, cardinalities=[], emb_dims=[], hidden=(8,))
    out = model(torch.randn(4, 5), torch.zeros((4, 0), dtype=torch.int64))
    assert out.shape == (4,)


@pytest.fixture(scope="module")
def trained():
    X, y = make_table(4000, seed=1)
    enc = TabularEncoder(CATS).fit(X.iloc[:3000])
    tr, va = enc.transform(X.iloc[:3000]), enc.transform(X.iloc[3000:])
    config = TrainConfig(hidden=(16,), max_epochs=15, patience=5, batch_size=256, lr=3e-3)
    model, history = train_model(tr, y[:3000], va, y[3000:], enc.cardinalities, config, seed=42)
    return model, history, enc, tr, va, y, config


def test_training_learns_the_signal(trained):
    model, history, _, _, va, y, _ = trained
    base_rate = y[3000:].mean()
    assert average_precision_score(y[3000:], predict_proba(model, va)) > base_rate + 0.2
    assert history.attrs["best_epoch"] >= 1
    assert {"train_loss", "val_loss", "val_pr_auc"} <= set(history.columns)


def test_keeps_best_epoch_weights(trained):
    model, history, _, _, va, y, _ = trained
    score = average_precision_score(y[3000:], predict_proba(model, va))
    assert score == pytest.approx(history["val_pr_auc"].max(), abs=1e-9)


def test_learning_rate_starts_at_config_and_only_goes_down(trained):
    _, history, _, _, _, _, config = trained
    assert history["lr"].iloc[0] == pytest.approx(config.lr)
    assert history["lr"].is_monotonic_decreasing  # halved on plateaus, never raised


def test_fixed_learning_rate_when_schedule_is_off(trained):
    _, _, enc, tr, va, y, config = trained
    fixed = TrainConfig(**{**config.__dict__, "lr_patience": 0, "max_epochs": 3})
    _, history = train_model(tr, y[:3000], va, y[3000:], enc.cardinalities, fixed, seed=42)
    assert (history["lr"] == config.lr).all()


def test_same_seed_same_model(trained):
    model, _, enc, tr, va, y, config = trained
    again, _ = train_model(tr, y[:3000], va, y[3000:], enc.cardinalities, config, seed=42)
    assert np.allclose(predict_proba(model, va), predict_proba(again, va))


def test_save_and_load_round_trip(trained, tmp_path):
    model, _, _, _, va, _, _ = trained
    path = tmp_path / "model.pt"
    save_model(model, path)
    assert np.allclose(predict_proba(model, va), predict_proba(load_model(path), va))


def test_sample_config_uses_space_and_fixed_values():
    rng = np.random.default_rng(0)
    space = {"dropout": [0.1, 0.3], "hidden": [(32,), (64, 32)]}
    config = sample_config(rng, space, max_epochs=7)
    assert config.dropout in (0.1, 0.3)
    assert config.hidden in ((32,), (64, 32))
    assert config.max_epochs == 7
