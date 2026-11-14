"""Tests for the modelling code. They run on the synthetic placeholder data."""

import numpy as np
import pandas as pd
import pytest

from shopsphere.baselines import recency_scores
from shopsphere.data import attach_labels, check_features, input_columns
from shopsphere.metrics import best_f1_threshold, evaluate, precision_at_top
from shopsphere.model import MLPParams
from shopsphere.placeholder import make_placeholder
from shopsphere.preprocess import build_preprocessor
from shopsphere.train import predict_proba, train_mlp


@pytest.fixture(scope="module")
def prepared():
    """Placeholder data taken through the same steps as the notebook."""
    features, labels = make_placeholder(n=3000)
    data = attach_labels(features, labels)
    numeric, categorical = input_columns(data)
    train, val = data[data["split"] == "train"], data[data["split"] == "val"]
    preprocessor = build_preprocessor(numeric, categorical).fit(train)
    return {
        "X_train": preprocessor.transform(train),
        "y_train": train["target"].to_numpy(),
        "X_val": preprocessor.transform(val),
        "y_val": val["target"].to_numpy(),
        "numeric": numeric,
        "categorical": categorical,
        "preprocessor": preprocessor,
    }


# --- data contract -------------------------------------------------------------------


def test_contract_rejects_target_column():
    features, labels = make_placeholder(n=100)
    with pytest.raises(ValueError, match="target"):
        check_features(features.assign(target=labels["target"]))


def test_contract_rejects_duplicate_customers():
    features, _ = make_placeholder(n=100)
    with pytest.raises(ValueError, match="duplicate"):
        check_features(pd.concat([features, features.head(1)]))


def test_contract_rejects_raw_dates():
    features, _ = make_placeholder(n=100)
    with pytest.raises(ValueError, match="dates"):
        check_features(features.assign(signup_date="2023-05-01"))


def test_model_inputs_exclude_id_target_split_and_noise(prepared):
    used = prepared["numeric"] + prepared["categorical"]
    assert not {"customer_id", "target", "split"} & set(used)
    assert not [c for c in used if c.startswith("noise_")]


# --- preprocessing -------------------------------------------------------------------


def test_preprocessed_matrix_is_complete_and_scaled(prepared):
    X = prepared["X_train"]
    assert not np.isnan(X).any()
    n_numeric = len(prepared["numeric"])
    assert X[:, :n_numeric].mean(axis=0) == pytest.approx(0, abs=1e-6)


def test_unseen_category_does_not_crash(prepared):
    features, _ = make_placeholder(n=5)
    features["country_code"] = "ZZ"
    assert not np.isnan(prepared["preprocessor"].transform(features)).any()


# --- metrics -------------------------------------------------------------------------


def test_evaluate_on_a_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1])
    result = evaluate(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]), threshold=0.5)
    assert result["pr_auc"] == result["roc_auc"] == result["f1"] == 1.0


def test_always_zero_gets_high_accuracy_but_finds_nobody():
    y = np.array([0] * 85 + [1] * 15)
    result = evaluate(y, np.zeros(100), threshold=0.5)
    assert result["accuracy"] == 0.85
    assert result["recall"] == 0.0
    assert result["pr_auc"] == pytest.approx(0.15)


def test_precision_at_top():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.arange(10, 0, -1)  # the two buyers are ranked first
    assert precision_at_top(y, score, 0.2) == 1.0
    assert precision_at_top(y, score, 0.5) == 0.4


def test_best_f1_threshold_separates_the_classes():
    y = np.array([0, 0, 0, 1, 1])
    threshold = best_f1_threshold(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]))
    assert 0.3 < threshold <= 0.8


def test_recency_ranks_recent_first_and_never_ordered_last():
    scores = recency_scores(pd.Series([5, 200, np.nan]))
    assert scores[0] > scores[1] > scores[2]


# --- training ------------------------------------------------------------------------


def test_mlp_learns_and_outputs_probabilities(prepared):
    params = MLPParams(max_epochs=15)
    model, history = train_mlp(
        prepared["X_train"], prepared["y_train"], prepared["X_val"], prepared["y_val"], params
    )
    proba = predict_proba(model, prepared["X_val"])
    assert ((proba >= 0) & (proba <= 1)).all()
    base_rate = prepared["y_val"].mean()
    assert history["val_pr_auc"].max() > 2 * base_rate  # far better than random


def test_same_seed_gives_same_model(prepared):
    params = MLPParams(max_epochs=3)
    args = (prepared["X_train"], prepared["y_train"], prepared["X_val"], prepared["y_val"], params)
    first, _ = train_mlp(*args, seed=7)
    second, _ = train_mlp(*args, seed=7)
    np.testing.assert_allclose(
        predict_proba(first, prepared["X_val"]), predict_proba(second, prepared["X_val"]), rtol=1e-5
    )
