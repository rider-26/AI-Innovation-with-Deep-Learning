"""Build the WEEKLY ACTIVITY table for the GRU/LSTM model (Person 3).

Why this file exists
    train_features.csv squeezes each customer's history into totals ("12 orders in the last
    year"). Totals hide the SHAPE of the history: a customer who was very active and is now
    slowing down can have the same total as one who is just getting started. A GRU reads the
    history week by week, so it can see that trend.

What it builds
    For every customer: 104 weeks counted backwards from the 30 Sep 2024 cutoff, with
    4 numbers per week: orders, spend, sessions, cart_adds.

        position 103 (the last one) = 24-30 Sep 2024  (the most recent week)
        position 102                = 17-23 Sep 2024
        ...

    Saved as data/processed/weekly_sequences.npz, an array of shape (customers, 104, 4)
    plus the matching customer_id list. The model notebook can use the last 26 or 52 weeks.

Leakage
    Only rows dated on or before CUTOFF_DATE (from shopsphere.config) are used, so no
    week can contain October data.

Run from the project root:
    python src/weekly_sequences.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the team package (src/shopsphere) importable, like src/data_prep.py does.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shopsphere.config import CUTOFF_DATE, DATA_PROCESSED, DATA_RAW, ID_COL  # noqa: E402

CUTOFF = pd.Timestamp(CUTOFF_DATE)
N_WEEKS = 104                       # two years of history; the notebook can use fewer
CHANNELS = ["orders", "spend", "sessions", "cart_adds"]
OUT_PATH = DATA_PROCESSED / "weekly_sequences.npz"


def load_transactions() -> pd.DataFrame:
    """Orders up to the cutoff, with the known data-quality fixes for the columns we use."""
    tx = pd.read_csv(DATA_RAW / "transactions.csv",
                     usecols=["transaction_id", ID_COL, "transaction_timestamp", "order_value"])
    tx = tx.drop_duplicates(subset="transaction_id", keep="first")    # 904 duplicate exports
    tx["order_value"] = pd.to_numeric(                                 # "$1,234.50" -> 1234.5
        tx["order_value"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    tx["date"] = pd.to_datetime(tx["transaction_timestamp"], errors="coerce")
    return tx[tx["date"] <= CUTOFF]                                    # leakage guard


def load_sessions() -> pd.DataFrame:
    """Website/app sessions up to the cutoff."""
    ss = pd.read_csv(DATA_RAW / "sessions.csv",
                     usecols=["session_id", ID_COL, "session_timestamp", "cart_additions"])
    ss = ss.drop_duplicates(subset="session_id", keep="first")
    ss["cart_additions"] = pd.to_numeric(ss["cart_additions"], errors="coerce").clip(lower=0)
    ss["date"] = pd.to_datetime(ss["session_timestamp"], errors="coerce")
    return ss[ss["date"] <= CUTOFF]                                    # leakage guard


def weeks_before_cutoff(dates: pd.Series) -> pd.Series:
    """0 = the 7 days ending on the cutoff (24-30 Sep), 1 = the 7 days before that, ..."""
    return ((CUTOFF - dates.dt.normalize()).dt.days // 7).astype(int)


def build(n_weeks: int = N_WEEKS):
    customers = np.sort(pd.read_csv(DATA_RAW / "customers.csv", usecols=[ID_COL])[ID_COL].unique())
    row_of = pd.Series(np.arange(len(customers)), index=customers)    # customer_id -> row number

    tx, ss = load_transactions(), load_sessions()
    tx["w"], ss["w"] = weeks_before_cutoff(tx["date"]), weeks_before_cutoff(ss["date"])
    tx, ss = tx[tx["w"] < n_weeks], ss[ss["w"] < n_weeks]

    # one row per (customer, week) with the 4 numbers
    orders = tx.groupby([ID_COL, "w"]).agg(orders=("transaction_id", "size"),
                                           spend=("order_value", "sum"))
    visits = ss.groupby([ID_COL, "w"]).agg(sessions=("session_id", "size"),
                                           cart_adds=("cart_additions", "sum"))
    weekly = orders.join(visits, how="outer").fillna(0).reset_index()
    weekly = weekly[weekly[ID_COL].isin(row_of.index)]

    # Dense 3-D array (customer, time step, channel). Oldest week first, newest LAST,
    # because a GRU reads left to right and its final state should reflect the latest weeks.
    seq = np.zeros((len(customers), n_weeks, len(CHANNELS)), dtype=np.float32)
    seq[row_of[weekly[ID_COL]].values, (n_weeks - 1) - weekly["w"].values] = \
        weekly[CHANNELS].values.astype(np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, seq=seq, customer_id=customers, channels=np.array(CHANNELS))
    print(f"Saved {OUT_PATH.name}: {seq.shape[0]:,} customers x {seq.shape[1]} weeks "
          f"x {seq.shape[2]} channels")
    print(f"Customer-weeks with any activity: {(seq.sum(axis=2) > 0).mean():.1%}")
    return seq, customers


if __name__ == "__main__":
    build()
