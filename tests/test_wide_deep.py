"""Unit tests for src/wide_deep.py. They use a small made-up table, so no data pack is needed.

Run from the project root:  python -m pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wide_deep import (
    NEVER_ORDERED_DAYS,
    PlattCalibrator,
    TrainConfig,
    WideAndDeep,
    WideFeatures,
    interaction_test,
    load_model,
    predict_proba,
    recency_bucket,
    train_model,
)


@pytest.fixture
def table():
    """600 made-up customers. Gold customers buy when recent; Bronze customers rarely buy."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({
        "customer_tier": rng.choice(["Bronze", "Gold"], n),
        "days_since_last_order": rng.choice([5, 45, 200, NEVER_ORDERED_DAYS], n),
        "spend_total": rng.gamma(2, 100, n),
        "never_ordered": 0,
    })
    X.loc[X["days_since_last_order"] == NEVER_ORDERED_DAYS, "never_ordered"] = 1
    recent = X["days_since_last_order"] <= 60
    p = np.where((X["customer_tier"] == "Gold") & recent, 0.8, 0.1)
    y = (rng.random(n) < p).astype(int)
    return X, y


def test_recency_buckets_edges():
    days = pd.Series([0, 30, 31, 60, 365, 366, 1099, NEVER_ORDERED_DAYS])
    assert list(recency_bucket(days)) == [
        "0-30", "0-30", "31-60", "31-60", "181-365", "366+", "366+", "never"]


def test_wide_features_shape_and_names(table):
    X, _ = table
    wf = WideFeatures([("customer_tier", "days_since_last_order")], min_count=1).fit(X)
    Z = wf.transform(X)
    names = wf.get_feature_names_out()
    assert Z.shape == (len(X), len(names))
    assert "flag: never_ordered" in names  # 0/1 column found automatically
    assert any("Gold | 0-30" in n for n in names)  # the cross exists
    # each cross column is one-hot: exactly one 1 per row in the cross block
    cross_cols = [i for i, n in enumerate(names) if " | " in n]
    assert np.all(Z[:, cross_cols].sum(axis=1) == 1)


def test_unseen_category_does_not_crash(table):
    X, _ = table
    wf = WideFeatures([("customer_tier", "days_since_last_order")]).fit(X)
    new = X.head(3).copy()
    new["customer_tier"] = "Platinum"  # never seen in "train"
    assert wf.transform(new).shape[0] == 3


def test_rare_crosses_are_merged(table):
    X, _ = table
    many = WideFeatures([("customer_tier", "spend_total")], min_count=1).fit(X)
    few = WideFeatures([("customer_tier", "spend_total")], min_count=10_000).fit(X)
    assert few.transform(X).shape[1] < many.transform(X).shape[1]


def test_interaction_test_finds_planted_interaction(table):
    X, y = table
    res = interaction_test(X, y, X, y, "customer_tier", "days_since_last_order")
    assert res["deviance_gain"] > 0
    assert res["p_value"] < 0.01


def test_ablation_switches():
    xw, xd = torch.ones(4, 3), torch.ones(4, 5)
    assert WideAndDeep(3, 5, use_wide=False)(xw, xd).shape == (4,)
    assert WideAndDeep(3, 5, use_deep=False).deep is None
    with pytest.raises(ValueError):
        WideAndDeep(3, 5, use_wide=False, use_deep=False)


def test_training_learns_and_reloads(table, tmp_path):
    X, y = table
    wf = WideFeatures([("customer_tier", "days_since_last_order")], min_count=1).fit(X)
    xw = wf.transform(X)
    xd = np.random.default_rng(1).normal(size=(len(X), 4)).astype(np.float32)  # noise only
    cfg = TrainConfig(hidden=(8,), max_epochs=40, patience=40, lr=1e-2)
    model, history = train_model((xw, xd), y, (xw, xd), y, cfg)
    assert history["val_pr_auc"].max() > 0.6  # the planted rule is easy to learn
    p = predict_proba(model, xw, xd)
    assert p.min() >= 0 and p.max() <= 1

    from wide_deep import save_model
    save_model(model, tmp_path / "m.pt")
    again = load_model(tmp_path / "m.pt")
    assert np.allclose(predict_proba(again, xw, xd), p)


def test_calibration_keeps_ranking():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=500)
    y = (rng.random(500) < 1 / (1 + np.exp(-2 * logits))).astype(int)
    cal = PlattCalibrator().fit(logits, y)
    p = cal.predict_proba(logits)
    assert np.all(np.diff(p[np.argsort(logits)]) >= -1e-12)  # same order as the logits
