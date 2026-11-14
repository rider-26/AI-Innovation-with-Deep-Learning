"""Load the feature table, check it against the team's data contract, and attach labels.

The contract (project plan, section 4.2): one row per customer, `customer_id` as the key,
no target column, no raw dates. Checking it here means a bad file fails immediately with a
clear message, instead of producing a model that silently trains on the wrong thing.
"""

from pathlib import Path

import pandas as pd

from shopsphere.config import DROP_COLS, ID_COL, TARGET_COL
from shopsphere.split import assign_split


def load_features(path: Path) -> pd.DataFrame:
    """Read features.csv and raise a clear error if it breaks the data contract."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Feature table not found: {path}")
    features = pd.read_csv(path)
    check_features(features)
    return features


def check_features(features: pd.DataFrame) -> None:
    """Raise ValueError describing the first contract violation found."""
    if ID_COL not in features.columns:
        raise ValueError(f"Feature table has no '{ID_COL}' column")
    if features[ID_COL].duplicated().any():
        n = int(features[ID_COL].duplicated().sum())
        raise ValueError(f"Feature table must have one row per customer: {n} duplicate ids")
    if TARGET_COL in features.columns:
        # The target inside the feature table is the most direct form of leakage.
        raise ValueError(f"Feature table must not contain the '{TARGET_COL}' column")

    # Raw dates must be converted to numbers first (e.g. days_since_last_order). Left as
    # text they would be one-hot encoded into thousands of meaningless columns.
    for col in features.columns.drop(ID_COL):
        if _is_text(features[col]):
            sample = features[col].dropna().head(50)
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if len(sample) > 0 and parsed.notna().all():
                raise ValueError(f"Column '{col}' looks like raw dates: convert it to a number")


def input_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split the model's input columns into (numeric, categorical) by their data type.

    Leaves out the id, the target, the split label, anything in DROP_COLS and the
    `noise_*` system fields, which the data dictionary says have no business meaning.
    """
    excluded = {ID_COL, TARGET_COL, "split", *DROP_COLS}
    candidates = [c for c in features.columns if c not in excluded and not c.startswith("noise_")]
    categorical = [c for c in candidates if _is_text(features[c])]
    numeric = [c for c in candidates if c not in categorical]
    return numeric, categorical


def _is_text(column: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)


def attach_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join the labelled customers onto their features and add the shared split column.

    Only labelled customers are returned (the 64,000 in train.csv). The 16,000 in test.csv
    have no target, so they cannot be used for training or for scoring the model.
    """
    missing = set(labels[ID_COL]) - set(features[ID_COL])
    if missing:
        raise ValueError(f"{len(missing)} labelled customers have no row in the feature table")
    labelled = assign_split(labels)
    return features.merge(labelled, on=ID_COL, how="inner", validate="one_to_one")
