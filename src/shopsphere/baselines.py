"""Non-deep-learning baselines the neural network has to beat.

Each function returns scores where higher means "more likely to buy", so all of them can
be passed to the same `metrics.evaluate`.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from shopsphere.config import SEED


def always_zero_scores(n: int) -> np.ndarray:
    """The floor: predict "won't buy" for everyone.

    Scores about 85% accuracy while finding no buyers at all, which is why accuracy alone
    cannot be the metric.
    """
    return np.zeros(n)


def recency_scores(days_since_last_order: pd.Series) -> np.ndarray:
    """What the CRM team does today: rank by how recently the customer ordered.

    Fewer days means a higher score. Customers with no value (never ordered) go last.
    There is nothing to fit, so this baseline cannot overfit.
    """
    days = pd.to_numeric(days_since_last_order, errors="coerce")
    return -days.fillna(days.max() + 1).to_numpy(dtype=float)


def fit_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """The simple ML baseline: one weight per input, no interactions between features.

    class_weight="balanced" makes a missed buyer cost more than a false alarm, in
    proportion to how rare buyers are. Without it the model leans towards "won't buy".
    """
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    return model.fit(X_train, y_train)


def fit_lightgbm(X_train: np.ndarray, y_train: np.ndarray) -> LGBMClassifier:
    """The strong ML baseline: gradient-boosted trees, usually the model to beat on tables."""
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=SEED,
        verbose=-1,
    )
    return model.fit(X_train, y_train)
