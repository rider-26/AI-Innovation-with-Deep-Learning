"""Shared data loading, split, preprocessing and metrics for the ShopSphere purchase model.

Every team member imports from this file so that all models (baselines and deep learning)
use exactly the same rows, the same preprocessing and the same metric definitions.

Usage (from a notebook in the project root or in notebooks/):

    import sys
    sys.path.insert(0, "src")          # or "../src" from inside notebooks/
    from data_prep import load_data, get_splits, split_xy, build_preprocessor
    from data_prep import best_f1_threshold, evaluate

    df = load_data()
    train, val, test = get_splits(df)
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)

    pre = build_preprocessor(X_train).fit(X_train)   # fit on TRAIN only
    X_val_ready = pre.transform(X_val)

    threshold = best_f1_threshold(y_val, val_scores)  # threshold chosen on VALIDATION
    metrics = evaluate(y_test, test_scores, threshold)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths: everything is relative to the project root, so it runs on any machine
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make the team package (src/shopsphere) importable even without `uv sync`.
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shopsphere.config import ID_COL, SEED, TARGET_COL  # noqa: E402
from shopsphere.split import assign_split  # noqa: E402


def _find_data_dir() -> Path:
    """Return the folder that holds train_features.csv.

    Some of us keep the feature files in `cleaned data/`, the repo layout keeps them in
    `data/processed/`. Use whichever one actually contains the file.
    """
    candidates = [PROJECT_ROOT / "cleaned data", PROJECT_ROOT / "data" / "processed"]
    for folder in candidates:
        if (folder / "train_features.csv").exists():
            return folder
    raise FileNotFoundError(
        "train_features.csv not found. Put it in one of: " + ", ".join(str(c) for c in candidates)
    )


DATA_DIR = _find_data_dir()
TRAIN_FEATURES_PATH = DATA_DIR / "train_features.csv"
SPLITS_PATH = DATA_DIR / "splits.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# The six text columns. Every other feature column is numeric.
CATEGORICAL_COLS = [
    "country_code",
    "region",
    "customer_tier",
    "region_tier",
    "preferred_channel",
    "preferred_language",
]


# ---------------------------------------------------------------------------
# Data and split
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the labelled feature table: one row per customer, features + `target`.

    Only train_features.csv is read. test_features.csv has no labels, so it is never
    used for training or evaluation.
    """
    return pd.read_csv(TRAIN_FEATURES_PATH)


def get_splits(df: pd.DataFrame | None = None):
    """Return (train, val, test) DataFrames using the one shared split.

    The assignment lives in `splits.csv` (columns: customer_id, split). If that file
    exists it is loaded, so everyone gets identical rows. If it does not exist it is
    created once: 70/15/15, stratified on the target, random_state=42
    (via shopsphere.split.assign_split, the team's single split function).
    """
    if df is None:
        df = load_data()

    if SPLITS_PATH.exists():
        splits = pd.read_csv(SPLITS_PATH)
    else:
        splits = assign_split(df[[ID_COL, TARGET_COL]])[[ID_COL, "split"]]
        splits.to_csv(SPLITS_PATH, index=False)

    # Safety checks: the split file must cover exactly the customers in the data.
    if splits[ID_COL].duplicated().any():
        raise ValueError(f"{SPLITS_PATH} has duplicate customer_id values")
    if set(splits[ID_COL]) != set(df[ID_COL]):
        raise ValueError(f"{SPLITS_PATH} does not match the customers in train_features.csv")

    merged = df.merge(splits, on=ID_COL, how="left", validate="one_to_one")
    parts = []
    for name in ["train", "val", "test"]:
        part = merged[merged["split"] == name].drop(columns="split").reset_index(drop=True)
        parts.append(part)
    return tuple(parts)


def split_xy(part: pd.DataFrame):
    """Split one part into X (features only) and y (target).

    customer_id is dropped here: it is a key, never a feature.
    """
    X = part.drop(columns=[ID_COL, TARGET_COL])
    y = part[TARGET_COL].to_numpy()
    return X, y


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Return an UNFITTED preprocessor for the feature table X.

    numeric columns:     median imputation -> standard scaling
    categorical columns: one-hot encoding (categories unseen in training become all zeros)

    Call .fit() on the TRAIN split only, then .transform() validation and test.
    X is only used to read the column names; no statistics are learned here.
    """
    categorical = [c for c in CATEGORICAL_COLS if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_steps, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        verbose_feature_names_out=False,  # keep readable names, e.g. "country_code_SG"
    )


# ---------------------------------------------------------------------------
# Metrics: identical for every model
# ---------------------------------------------------------------------------
def best_f1_threshold(y_true, scores) -> float:
    """Return the score threshold that maximises F1. Call this on VALIDATION only.

    A customer is predicted as a buyer when score >= threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall pair has no threshold, so drop it.
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def precision_at_top(y_true, scores, fraction: float = 0.10) -> float:
    """Share of real buyers among the top `fraction` of customers ranked by score.

    When several customers share the score at the cut-off (for example many customers
    with the same days_since_last_order), the tied group is counted at its average buy
    rate. This is the expected result of breaking ties at random, and it makes the
    number independent of the row order.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    k = int(np.ceil(fraction * len(scores)))

    cutoff = np.sort(scores)[::-1][k - 1]  # score of the k-th best customer
    above = scores > cutoff
    tied = scores == cutoff
    buyers = y_true[above].sum() + (k - above.sum()) * y_true[tied].mean()
    return float(buyers / k)


def evaluate(y_true, scores, threshold: float) -> dict:
    """Compute every reported metric for one model on one split.

    y_true:    0/1 labels
    scores:    higher = more likely to buy (probabilities or any ranking score)
    threshold: a customer is predicted as a buyer when score >= threshold.
               Choose it on validation with best_f1_threshold(), then reuse it on test.

    PR-AUC, ROC-AUC, precision@top10% and lift only depend on the ranking.
    Precision, recall, F1 and the confusion matrix depend on the threshold.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    base_rate = float(y_true.mean())
    top10 = precision_at_top(y_true, scores, 0.10)

    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision_top10": top10,
        "lift_top10": top10 / base_rate,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "SPLITS_PATH",
    "MODELS_DIR",
    "RESULTS_DIR",
    "CATEGORICAL_COLS",
    "ID_COL",
    "TARGET_COL",
    "SEED",
    "load_data",
    "get_splits",
    "split_xy",
    "build_preprocessor",
    "best_f1_threshold",
    "precision_at_top",
    "evaluate",
]
