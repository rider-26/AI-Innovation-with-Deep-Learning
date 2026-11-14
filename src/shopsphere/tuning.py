"""Hyperparameter search for the MLP, with every run written to a log.

The log is the evidence of optimisation: it shows what was tried, in what order, and what
each attempt scored on validation.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from shopsphere.config import SEED
from shopsphere.model import MLPParams
from shopsphere.train import train_mlp

# The values each setting may take. Random search draws one value per setting per trial.
SEARCH_SPACE = {
    "hidden": [(64,), (128, 64), (256, 128), (256, 128, 64)],
    "dropout": [0.1, 0.2, 0.3, 0.5],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-5, 1e-4],
    "batch_size": [256, 512, 1024],
}


def random_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    log_path: Path,
    seed: int = SEED,
) -> pd.DataFrame:
    """Train `n_trials` MLPs with randomly drawn settings and return the log, best first.

    Trial 0 is always the default configuration from the project plan, so every other
    trial can be read as "better or worse than where we started".

    Models are compared on VALIDATION PR-AUC. The test split is never used here.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(n_trials):
        if trial == 0:
            params = MLPParams()
        else:
            params = MLPParams(
                **{name: values[rng.integers(len(values))] for name, values in SEARCH_SPACE.items()}
            )

        started = time.perf_counter()
        _, history = train_mlp(X_train, y_train, X_val, y_val, params, seed=seed)
        best = history.loc[history["val_pr_auc"].idxmax()]

        rows.append(
            {
                "trial": trial,
                **{name: getattr(params, name) for name in SEARCH_SPACE},
                "best_epoch": int(best["epoch"]),
                "epochs_run": len(history),
                "train_pr_auc": best["train_pr_auc"],
                "val_pr_auc": best["val_pr_auc"],
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        # Written after every trial, so an interrupted search still leaves a record.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(log_path, index=False)

    return pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)


def params_from_row(row: pd.Series) -> MLPParams:
    """Rebuild the MLPParams of one row of the tuning log."""
    return MLPParams(
        hidden=tuple(row["hidden"]),
        dropout=float(row["dropout"]),
        lr=float(row["lr"]),
        weight_decay=float(row["weight_decay"]),
        batch_size=int(row["batch_size"]),
    )
