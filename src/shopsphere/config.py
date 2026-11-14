"""Team-wide constants. Every notebook and module imports these instead of redefining them,
so all four members train and evaluate on exactly the same data."""

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"  # the data pack, read-only
DATA_INTERIM = ROOT / "data" / "interim"  # cleaned tables
DATA_PROCESSED = ROOT / "data" / "processed"  # features.csv, split assignments
DATA_SYNTHETIC = ROOT / "data" / "synthetic"  # anything we generate, labelled as synthetic
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

ID_COL = "customer_id"
TARGET_COL = "target"

SEED = 42

# Leakage boundary: features may only use data dated on or before the cutoff.
CUTOFF_DATE = date(2024, 9, 30)
# The target is "at least one purchase in this window".
TARGET_WINDOW = (date(2024, 10, 1), date(2024, 10, 30))

# Train / validation / test fractions of the labelled customers in train.csv.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------------------------------
# Feature table. EDIT THIS SECTION when the real features.csv arrives: it is the only place
# that names specific columns. Everything else detects column types automatically.
# ---------------------------------------------------------------------------------------
FEATURES_FILE = DATA_PROCESSED / "features.csv"  # one row per customer, no target column
LABELS_FILE = DATA_RAW / "train.csv"  # customer_id, target

# The recency baseline ranks customers by this column (fewer days = more likely to buy).
RECENCY_COL = "days_since_last_order"

# Columns that are in the file but must never be model inputs.
# Columns starting with "noise_" are dropped automatically as well.
DROP_COLS: list[str] = []

# Fairness check: metrics are reported separately for each value of these columns.
GROUP_COLS = ["country_code", "customer_tier"]
