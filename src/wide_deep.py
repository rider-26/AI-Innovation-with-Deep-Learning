"""Wide-and-deep model for the ShopSphere purchase model (Person 4: Jadon).

Same job as the baselines in 01_baselines.ipynb: give each customer the probability that they
buy at least once between 1 and 30 October 2024.

The idea (Cheng et al., 2016, "Wide & Deep Learning for Recommender Systems"):

    WIDE part = a linear model (like logistic regression) on simple yes/no features and
                CROSSED features such as "tier = Gold AND last order 31-60 days ago".
                It MEMORISES rules: one weight per rule.
    DEEP part = an MLP on the same inputs the other members use (scaled numbers + one-hot).
                It GENERALISES: it finds smooth, complex patterns.

    customer_tier x recency bucket ---+
    country x channel ----------------+-- one-hot --> Linear(n_wide, 1) ------------+
    0/1 flags ------------------------+                                             |
                                                                                    +--> add --> sigmoid
    numeric (impute + scale) ---------+                                             |
    text columns (one-hot) -----------+-- [Dense -> ReLU -> Dropout] x k -> Dense(1)+

Both parts are trained TOGETHER with one loss, so the deep part only has to learn what the
rules in the wide part cannot already explain.

The question this model answers: does adding memorised rules help the neural network?
We answer it with an ablation: deep only vs wide only vs wide + deep, same split, same seed.

Everything that learns from data (bucket edges, the list of crosses, medians, means, weights)
is fitted on the TRAIN split only. Validation is used for early stopping and choosing settings.
The test split is never touched in this file.

This module does not import data_prep.py, so it can be tested without the data pack.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from torch import nn

# ---------------------------------------------------------------------------
# Buckets: turn a number into a small set of labelled ranges
# ---------------------------------------------------------------------------
# Recency gets business-meaningful buckets instead of quantiles, so a rule reads like
# "last order 31-60 days ago". 1100 days is the code for "never ordered" in features.csv.
RECENCY_COL = "days_since_last_order"
NEVER_ORDERED_DAYS = 1100
RECENCY_EDGES = [30, 60, 90, 180, 365, NEVER_ORDERED_DAYS - 1]
RECENCY_LABELS = ["0-30", "31-60", "61-90", "91-180", "181-365", "366+", "never"]

CROSS_SEP = " x "  # how a cross is written in names and logs, e.g. "customer_tier x region_tier"


def recency_bucket(days: pd.Series) -> pd.Series:
    """Days since last order -> one of RECENCY_LABELS. Bucket upper edges are inclusive."""
    edges = [-np.inf, *RECENCY_EDGES, np.inf]
    return pd.cut(days.astype(float), bins=edges, labels=RECENCY_LABELS, right=True).astype(str)


class Bucketizer:
    """Turn each numeric column into text labels for its range ("q1".."q5" or recency buckets).

    The edges are learned from TRAIN only:
      - days_since_last_order: the fixed business buckets above;
      - a column with 6 or fewer distinct values (a flag or a small count): each value is its
        own bucket;
      - any other number: quantile buckets (default 5), so each bucket holds about the same
        number of train customers.
    Text columns pass through unchanged. Missing numbers get their own bucket, "missing".
    """

    def __init__(self, n_buckets: int = 5):
        self.n_buckets = n_buckets

    def fit(self, X: pd.DataFrame, columns: list[str]) -> Bucketizer:
        self.columns_ = list(columns)
        self.edges_ = {}
        for col in self.columns_:
            s = X[col]
            if col == RECENCY_COL or not pd.api.types.is_numeric_dtype(s):
                continue
            values = s.dropna().unique()
            if len(values) <= 6:
                self.edges_[col] = None  # small set of values: use them as they are
            else:
                qs = np.linspace(0, 1, self.n_buckets + 1)[1:-1]
                self.edges_[col] = np.unique(np.quantile(s.dropna(), qs))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.columns_:
            s = X[col]
            if col == RECENCY_COL:
                out[col] = recency_bucket(s)
            elif not pd.api.types.is_numeric_dtype(s):
                out[col] = s.astype(str)
            elif self.edges_[col] is None:
                out[col] = s.map(lambda v: "missing" if pd.isna(v) else f"{v:g}")
            else:
                # np.searchsorted gives 0..n_buckets-1: which range the value falls in
                idx = np.searchsorted(self.edges_[col], s.to_numpy(dtype=float), side="right")
                labels = pd.Series([f"q{i + 1}" for i in idx], index=s.index)
                out[col] = labels.where(s.notna(), "missing")
        return pd.DataFrame(out, index=X.index)


# ---------------------------------------------------------------------------
# Inputs for the WIDE part
# ---------------------------------------------------------------------------
def find_flags(X: pd.DataFrame) -> list[str]:
    """Numeric columns that only hold 0 and 1 (for example never_ordered)."""
    return [
        c
        for c in X.select_dtypes("number").columns
        if set(pd.unique(X[c].dropna())) <= {0, 1} and X[c].nunique() == 2
    ]


class WideFeatures:
    """Build the wide part's input: one-hot crossed features + 0/1 flags (+ optional main effects).

    crosses:  list of column pairs, e.g. [("customer_tier", "days_since_last_order")].
              Each pair becomes ONE text column whose value is the combination,
              e.g. "Gold | 31-60". One-hot encoding then gives one 0/1 input per combination,
              so the linear layer learns one weight (one "rule") per combination.
    flags:    0/1 columns passed through as they are. "auto" = every 0/1 column in train.
    main_effects: also one-hot each bucketed column on its own (tier alone, recency alone).
    min_count: a combination seen fewer than this many times in TRAIN is merged into an
               "infrequent" bucket. A rule learned from 5 customers is noise, not memory.
    """

    def __init__(self, crosses, flags="auto", main_effects: bool = True, min_count: int = 30,
                 n_buckets: int = 5):
        self.crosses = [tuple(c) for c in crosses]
        self.flags = flags
        self.main_effects = main_effects
        self.min_count = min_count
        self.n_buckets = n_buckets

    def _text_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Bucketed columns + one combined text column per cross."""
        b = self.bucketizer_.transform(X)
        parts = {}
        if self.main_effects:
            for col in self.base_cols_:
                parts[col] = b[col]
        for a, c in self.crosses:
            parts[f"{a}{CROSS_SEP}{c}"] = b[a] + " | " + b[c]
        return pd.DataFrame(parts, index=X.index)

    def fit(self, X: pd.DataFrame) -> WideFeatures:
        self.base_cols_ = sorted({col for pair in self.crosses for col in pair})
        missing = [c for c in self.base_cols_ if c not in X.columns]
        if missing:
            raise KeyError(f"cross columns not in the table: {missing}")
        self.flag_cols_ = find_flags(X) if self.flags == "auto" else list(self.flags)
        self.bucketizer_ = Bucketizer(self.n_buckets).fit(X, self.base_cols_)
        text = self._text_columns(X)
        self.encoder_ = None
        if text.shape[1] > 0:
            self.encoder_ = OneHotEncoder(
                handle_unknown="infrequent_if_exist",  # unseen combination -> "infrequent"
                min_frequency=self.min_count,
                sparse_output=False,
                dtype=np.float32,
            ).fit(text)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.encoder_ is not None:
            parts.append(self.encoder_.transform(self._text_columns(X)))
        if self.flag_cols_:
            parts.append(X[self.flag_cols_].fillna(0).to_numpy(np.float32))
        if not parts:
            return np.zeros((len(X), 0), dtype=np.float32)
        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> list[str]:
        names = list(self.encoder_.get_feature_names_out()) if self.encoder_ is not None else []
        return names + [f"flag: {c}" for c in self.flag_cols_]


# ---------------------------------------------------------------------------
# EDA helper: is an interaction real?
# ---------------------------------------------------------------------------
def interaction_test(X_train, y_train, X_val, y_val, col_a, col_b, n_buckets: int = 5) -> dict:
    """Does the COMBINATION of col_a and col_b tell us more than each column on its own?

    Two logistic regressions on bucketed versions of the two columns:
      main effects:  buy ~ a + b              (each column adds its own amount)
      with cross:    buy ~ a + b + a x b      (each combination can have its own rate)

    Returns
      deviance_gain: how much better the cross model fits TRAIN (2 x log-likelihood gain).
                     Under "no interaction" it follows a chi-squared distribution with
                     `extra_params` degrees of freedom, which gives `p_value`.
      val_logloss_gain: how much the cross lowers log loss on VALIDATION. With 45,000 rows
                     almost any p-value is tiny, so this is the practical test: does the
                     rule still help on customers the model has not seen?
    """
    bk = Bucketizer(n_buckets).fit(X_train, [col_a, col_b])
    bt, bv = bk.transform(X_train), bk.transform(X_val)

    def design(b, with_cross):
        cols = {"a": b[col_a], "b": b[col_b]}
        if with_cross:
            cols["ab"] = b[col_a] + " | " + b[col_b]
        return pd.DataFrame(cols)

    results = {}
    for with_cross in (False, True):
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=30, drop="first",
                            sparse_output=False)
        Zt = enc.fit_transform(design(bt, with_cross))
        Zv = enc.transform(design(bv, with_cross))
        # Large C = almost no regularisation, so the likelihood test is close to textbook.
        model = LogisticRegression(C=1e4, max_iter=2000).fit(Zt, y_train)
        results[with_cross] = {
            "n_params": Zt.shape[1],
            "train_ll": -log_loss(y_train, model.predict_proba(Zt)[:, 1], normalize=False),
            "val_logloss": log_loss(y_val, model.predict_proba(Zv)[:, 1]),
        }
    extra = max(results[True]["n_params"] - results[False]["n_params"], 1)
    gain = 2 * (results[True]["train_ll"] - results[False]["train_ll"])
    return {
        "cross": f"{col_a}{CROSS_SEP}{col_b}",
        "col_a": col_a,
        "col_b": col_b,
        "extra_params": extra,
        "deviance_gain": gain,
        "p_value": float(chi2.sf(max(gain, 0.0), extra)),
        "val_logloss_gain": results[False]["val_logloss"] - results[True]["val_logloss"],
    }


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
class WideAndDeep(nn.Module):
    """logit = wide(x_wide) + deep(x_deep). Turn off either part to get the ablation models.

    use_wide=False -> a plain MLP (the deep part alone)
    use_deep=False -> logistic regression on the wide features (the wide part alone)
    """

    def __init__(self, n_wide: int, n_deep: int, hidden=(64, 32), dropout: float = 0.2,
                 use_wide: bool = True, use_deep: bool = True):
        super().__init__()
        if not (use_wide or use_deep):
            raise ValueError("at least one of use_wide / use_deep must be True")
        self.init_kwargs = {"n_wide": int(n_wide), "n_deep": int(n_deep),
                            "hidden": [int(h) for h in hidden], "dropout": float(dropout),
                            "use_wide": bool(use_wide), "use_deep": bool(use_deep)}
        self.use_wide = use_wide and n_wide > 0
        self.use_deep = use_deep

        # Wide: one weight per rule + a bias. This IS logistic regression.
        self.wide = nn.Linear(n_wide, 1) if self.use_wide else None

        # Deep: Dense -> ReLU -> Dropout, repeated, then one output.
        if self.use_deep:
            layers, width = [], n_deep
            for units in hidden:
                layers += [nn.Linear(width, units), nn.ReLU(), nn.Dropout(dropout)]
                width = units
            layers.append(nn.Linear(width, 1))
            self.deep = nn.Sequential(*layers)
        else:
            self.deep = None

    def forward(self, x_wide: torch.Tensor, x_deep: torch.Tensor) -> torch.Tensor:
        """Return one logit per customer (sigmoid is applied in the loss / predict_proba)."""
        logit = torch.zeros(x_deep.shape[0], device=x_deep.device)
        if self.use_wide:
            logit = logit + self.wide(x_wide).squeeze(1)
        if self.use_deep:
            logit = logit + self.deep(x_deep).squeeze(1)
        return logit


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Every setting that can be tuned. The defaults are a reasonable first try."""

    hidden: tuple[int, ...] = (64, 32)  # units per Dense layer in the deep part
    dropout: float = 0.2
    lr: float = 1e-3  # Adam learning rate (both parts)
    wide_l2: float = 1e-4  # weight decay on the wide part only: shrinks rules with little data
    deep_l2: float = 1e-5  # weight decay on the deep part
    batch_size: int = 512
    pos_weight: float = 1.0  # loss weight on buyers (5.85 = "balanced")
    max_epochs: int = 100
    patience: int = 8  # stop after this many epochs without a better validation PR-AUC
    use_wide: bool = True
    use_deep: bool = True

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = "-".join(str(h) for h in self.hidden)
        return d


def set_seed(seed: int) -> None:
    """Fix every random number generator the training uses."""
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_logit(model: WideAndDeep, x_wide, x_deep, batch_size: int = 8192) -> np.ndarray:
    model.eval()  # dropout off
    xw = torch.as_tensor(x_wide, dtype=torch.float32)
    xd = torch.as_tensor(x_deep, dtype=torch.float32)
    out = [model(xw[i:i + batch_size], xd[i:i + batch_size]) for i in range(0, len(xd), batch_size)]
    return torch.cat(out).numpy()


def predict_proba(model: WideAndDeep, x_wide, x_deep) -> np.ndarray:
    """Probability of buying (before calibration)."""
    return 1 / (1 + np.exp(-predict_logit(model, x_wide, x_deep)))


def train_model(train_inputs, y_train, val_inputs, y_val, config: TrainConfig, seed: int = 42):
    """Train one model with early stopping on validation PR-AUC.

    train_inputs, val_inputs: (x_wide, x_deep) float32 matrices
    Returns (model, history). The model has the weights of its BEST validation epoch.
    """
    set_seed(seed)
    xw = torch.as_tensor(train_inputs[0], dtype=torch.float32)
    xd = torch.as_tensor(train_inputs[1], dtype=torch.float32)
    y = torch.as_tensor(np.array(y_train, dtype=np.float32))
    y_val = np.array(y_val, dtype=np.float32)  # a writable copy (pandas 3 labels are read-only)

    model = WideAndDeep(xw.shape[1], xd.shape[1], config.hidden, config.dropout,
                        config.use_wide, config.use_deep)

    # Two parameter groups, so the wide rules and the deep weights get their own penalty.
    groups = []
    if model.use_wide:
        groups.append({"params": model.wide.parameters(), "weight_decay": config.wide_l2})
    if model.use_deep:
        groups.append({"params": model.deep.parameters(), "weight_decay": config.deep_l2})
    optimiser = torch.optim.AdamW(groups, lr=config.lr)

    # pos_weight > 1 makes a missed buyer cost more. It also pushes probabilities up, which is
    # why the notebook calibrates the final model on validation.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(config.pos_weight)))
    val_loss_fn = nn.BCEWithLogitsLoss()

    generator = torch.Generator().manual_seed(seed)  # same batch order every run
    best_score, best_state, best_epoch, waited = -np.inf, None, 0, 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        start = time.perf_counter()
        model.train()  # dropout on
        order = torch.randperm(len(y), generator=generator)
        total = 0.0
        for i in range(0, len(y), config.batch_size):
            idx = order[i:i + config.batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(xw[idx], xd[idx]), y[idx])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)

        val_logit = predict_logit(model, *val_inputs)
        val_pr_auc = average_precision_score(y_val, val_logit)
        val_loss = val_loss_fn(torch.as_tensor(val_logit),
                               torch.as_tensor(y_val)).item()
        history.append({"epoch": epoch, "train_loss": total / len(y), "val_loss": val_loss,
                        "val_pr_auc": val_pr_auc, "seconds": time.perf_counter() - start})

        if val_pr_auc > best_score:  # keep a copy of the best weights so far
            best_score, best_epoch, waited = val_pr_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= config.patience:
                break

    model.load_state_dict(best_state)
    model.best_epoch = best_epoch
    return model, pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Calibration: make 0.7 mean "about 70% buy"
# ---------------------------------------------------------------------------
class PlattCalibrator:
    """Platt scaling: a 1-input logistic regression from the model's logit to a probability.

    Fitted on VALIDATION (never on train: the model's train scores are over-confident, and
    never on test). It only rescales the scores, so the ranking, PR-AUC and ROC-AUC do not change.
    """

    def fit(self, logits, y) -> PlattCalibrator:
        self.lr_ = LogisticRegression(C=1e4).fit(np.asarray(logits).reshape(-1, 1), y)
        return self

    def predict_proba(self, logits) -> np.ndarray:
        return self.lr_.predict_proba(np.asarray(logits).reshape(-1, 1))[:, 1]


def wide_rules(model: WideAndDeep, feature_names: list[str]) -> pd.Series:
    """The wide part's weight for every rule, sorted. Positive = pushes towards buying."""
    if not model.use_wide:
        return pd.Series(dtype=float)
    w = model.wide.weight.detach().numpy().ravel()
    return pd.Series(w, index=feature_names).sort_values()


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------
def save_model(model: WideAndDeep, path) -> None:
    torch.save({"init_kwargs": model.init_kwargs, "state_dict": model.state_dict()}, path)


def load_model(path) -> WideAndDeep:
    saved = torch.load(path, weights_only=True)
    model = WideAndDeep(**saved["init_kwargs"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model
