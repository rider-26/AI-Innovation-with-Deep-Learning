"""Turn the feature table into the all-numeric, scaled matrix a neural network needs.

Everything here is *fitted on the training split only* and then applied unchanged to
validation and test. Fitting on all rows would let validation and test statistics (their
medians, means and category lists) leak into training.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Return an unfitted preprocessor for the given numeric and categorical columns.

    Numeric: fill gaps with the median (robust to outliers), record that a value was missing
    (add_indicator), then standardise to mean 0 and standard deviation 1. A network trains
    poorly when inputs are on very different scales, e.g. days in the hundreds next to 0/1 flags.

    Categorical: fill gaps with the label "missing", then one-hot encode. A category seen
    only at prediction time is encoded as all zeros instead of raising an error.
    """
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_steps, numeric), ("cat", categorical_steps, categorical)],
        remainder="drop",  # any column not listed is not a model input
    )
