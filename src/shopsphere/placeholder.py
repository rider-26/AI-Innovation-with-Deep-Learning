"""SYNTHETIC stand-in for features.csv and train.csv.

This exists only so the modelling code can be run and tested before data preparation is
finished. It is NOT ShopSphere data, and no number produced from it may be reported.
The column names follow the examples in the project plan; the real file may differ.
"""

import numpy as np
import pandas as pd

from shopsphere.config import ID_COL, SEED, TARGET_COL


def make_placeholder(n: int = 8000, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, labels) shaped like the real files, with about 14.6% buyers."""
    rng = np.random.default_rng(seed)

    never_ordered = rng.random(n) < 0.15
    days_since_last_order = rng.exponential(120, n).round()
    orders_90d = rng.poisson(0.8, n)
    sessions_30d = rng.poisson(2.0, n)
    avg_order_value = rng.gamma(4, 20, n).round(2)
    late_deliveries_60d = rng.poisson(0.2, n)
    tier = rng.choice(["Bronze", "Silver", "Gold", "Platinum"], n, p=[0.5, 0.3, 0.15, 0.05])

    # A made-up relationship, so the models have something to learn.
    signal = (
        -0.012 * days_since_last_order
        + 0.5 * orders_90d
        + 0.25 * sessions_30d
        - 0.4 * late_deliveries_60d
        + pd.Series(tier).map({"Bronze": 0, "Silver": 0.3, "Gold": 0.6, "Platinum": 0.9}).to_numpy()
        - 2.0 * never_ordered
        + rng.normal(0, 1, n)
    )
    target = (signal > np.quantile(signal, 1 - 0.146)).astype(int)

    features = pd.DataFrame(
        {
            ID_COL: [f"SYN{i:06d}" for i in range(n)],
            "days_since_last_order": days_since_last_order,
            "never_ordered": never_ordered.astype(int),
            "orders_90d": orders_90d,
            "sessions_30d": sessions_30d,
            "avg_order_value": avg_order_value,
            "late_deliveries_60d": late_deliveries_60d,
            "country_code": rng.choice(["US", "UK", "DE", "FR", "IN"], n),
            "customer_tier": tier,
            "preferred_channel": rng.choice(["Email", "SMS", "WhatsApp"], n),
            "noise_customer_0": rng.normal(0, 1, n),
        }
    )
    # Real data has gaps: customers who never ordered have no recency or order value.
    features.loc[never_ordered, ["days_since_last_order", "avg_order_value"]] = np.nan
    features.loc[rng.random(n) < 0.03, "preferred_channel"] = np.nan

    labels = pd.DataFrame({ID_COL: features[ID_COL], TARGET_COL: target})
    return features, labels
