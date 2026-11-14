"""Evaluation metrics, chosen for a problem where only about 14.6% of customers buy.

Every model, baseline or neural network, is scored by the same functions, so the
comparison table is fair.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true, y_score) -> float:
    """Return the score cut-off that gives the highest F1.

    Call this on the VALIDATION split only. Choosing the threshold on the test split would
    tune the model to the test data and make the reported numbers too optimistic.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision and recall have one more entry than thresholds; the last has no threshold.
    return float(thresholds[np.argmax(f1[:-1])])


def precision_at_top(y_true, y_score, fraction: float = 0.10) -> float:
    """Share of real buyers among the top `fraction` of customers ranked by score.

    This is the business view: the retention budget only covers the top of the list.
    """
    y_true = np.asarray(y_true)
    k = max(1, round(len(y_true) * fraction))
    top = np.argsort(-np.asarray(y_score), kind="stable")[:k]
    return float(y_true[top].mean())


def evaluate(y_true, y_score, threshold: float) -> dict[str, float]:
    """Score one model on one split.

    y_score is any number where higher means "more likely to buy". PR-AUC, ROC-AUC and
    precision-at-top only use the ranking; the rest need the yes/no decision made by
    comparing the score to `threshold`.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    base_rate = float(y_true.mean())
    top10 = precision_at_top(y_true, y_score, 0.10)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),  # main metric
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_top10": top10,
        "lift_top10": top10 / base_rate,  # how many times better than picking at random
        "accuracy": float(accuracy_score(y_true, y_pred)),  # read next to 1 - base_rate
    }


def evaluate_by_group(y_true, y_score, groups, threshold: float) -> pd.DataFrame:
    """Run `evaluate` separately for each group (e.g. each country): the fairness check."""
    frame = pd.DataFrame(
        {"y": np.asarray(y_true), "score": np.asarray(y_score), "group": np.asarray(groups)}
    )
    rows = {}
    for name, part in frame.groupby("group"):
        if part["y"].nunique() < 2:
            continue  # ranking metrics are undefined when a group has only one class
        rows[name] = {
            "customers": len(part),
            "buy_rate": part["y"].mean(),
            **evaluate(part["y"], part["score"], threshold),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
