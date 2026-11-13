"""The one shared train / validation / test split.

Usage:
    labels = pd.read_csv(DATA_RAW / "train.csv")
    labels = assign_split(labels)
    train = labels[labels["split"] == "train"]
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from shopsphere.config import ID_COL, SEED, SPLIT_FRACTIONS, TARGET_COL


def assign_split(labels: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `labels` with a `split` column: "train", "val" or "test".

    Stratified on the target, so each part keeps the same share of buyers. Rows are
    sorted by customer_id first, so the result does not depend on the input row order.
    """
    if labels[ID_COL].duplicated().any():
        raise ValueError(f"{ID_COL} must be unique: de-duplicate before splitting")

    ordered = labels.sort_values(ID_COL).reset_index(drop=True)

    holdout_fraction = SPLIT_FRACTIONS["val"] + SPLIT_FRACTIONS["test"]
    train_ids, holdout = train_test_split(
        ordered[[ID_COL, TARGET_COL]],
        test_size=holdout_fraction,
        stratify=ordered[TARGET_COL],
        random_state=SEED,
    )
    val_ids, test_ids = train_test_split(
        holdout,
        test_size=SPLIT_FRACTIONS["test"] / holdout_fraction,
        stratify=holdout[TARGET_COL],
        random_state=SEED,
    )

    split_of = {
        **dict.fromkeys(train_ids[ID_COL], "train"),
        **dict.fromkeys(val_ids[ID_COL], "val"),
        **dict.fromkeys(test_ids[ID_COL], "test"),
    }
    out = labels.copy()
    out["split"] = out[ID_COL].map(split_of)
    return out
