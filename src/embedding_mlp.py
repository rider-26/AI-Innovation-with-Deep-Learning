"""MLP with learned category embeddings for the ShopSphere purchase model.

Same job as the baselines in 01_baselines.ipynb: give each customer a score, the probability
that they buy at least once between 1 and 30 October 2024. The difference from a plain MLP is
how the six text columns enter the network:

    plain MLP:   country_code = "SG"  ->  one-hot           [0, 0, 1, 0, 0]
    this model:  country_code = "SG"  ->  ID 3  ->  learned  [0.42, -1.10, 0.07]

Each text column gets its own embedding table: one small vector of numbers per category. The
vectors start random and are trained together with the rest of the network, so categories that
behave alike can end up with similar vectors.

    numeric columns -- log tails + impute + scale ----------+
    country_code ----- ID -- Embedding(levels + 1, d1) -----+
    ...                                                     +-- concat -> [Dense -> ReLU -> Dropout] x k
    region ----------- ID -- Embedding(levels + 1, d6) -----+          -> Dense(1) -> sigmoid

Everything that learns from data (medians, means, the list of categories, the network weights)
is fitted on the TRAIN split only. Validation is used for early stopping and for choosing
settings. The test split is never touched in this file.

This module does not import data_prep.py, so it can be tested without the data pack.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

# ID 0 is kept for "category not seen in the train split". Its embedding is fixed at zeros,
# which is the same thing OneHotEncoder(handle_unknown="ignore") does with an all-zero row.
UNKNOWN_ID = 0


# ---------------------------------------------------------------------------
# Preprocessing: feature table -> (numeric matrix, category ID matrix)
# ---------------------------------------------------------------------------
class TabularEncoder:
    """Turn the feature table into the two inputs the network needs.

    numeric columns:     log1p on long-tailed columns -> median imputation -> standard scaling
                         -> float32 matrix
    categorical columns: one integer ID per category seen in TRAIN (1, 2, 3, ...) -> int64 matrix;
                         a category never seen in train gets UNKNOWN_ID (0)

    Why the log step: columns like spend_total or credit_limit have a long right tail. After
    plain scaling, a few huge customers sit at +20 while everyone else is squeezed near 0, and a
    neural network learns badly from that. log1p(x) = log(1 + x) pulls the tail in and keeps the
    order of the customers. Trees (LightGBM) do not care about this; networks do.

    Call fit() on the train split only, then transform() train, validation and test.
    """

    def __init__(self, categorical_cols, log_skewed: bool = True, skew_threshold: float = 2.0):
        self.categorical_cols = list(categorical_cols)
        self.log_skewed = log_skewed
        self.skew_threshold = skew_threshold

    def _find_skewed(self, X_num: pd.DataFrame) -> list[str]:
        """Columns to log: never negative in train, more than two values (so not a 0/1 flag),
        and a right tail longer than `skew_threshold` (pandas skewness)."""
        return [
            c
            for c in X_num.columns
            if X_num[c].min() >= 0
            and X_num[c].nunique() > 2
            and X_num[c].skew() > self.skew_threshold
        ]

    def _log(self, X_num: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p to the columns chosen in fit(). Missing values stay missing for the imputer;
        clip(lower=0) guards against a negative value that never appeared in train."""
        if not self.log_cols_:
            return X_num
        X_num = X_num.copy()
        X_num[self.log_cols_] = np.log1p(X_num[self.log_cols_].astype(float).clip(lower=0))
        return X_num

    def fit(self, X: pd.DataFrame) -> TabularEncoder:
        missing = [c for c in self.categorical_cols if c not in X.columns]
        if missing:
            raise KeyError(f"categorical columns not in the table: {missing}")

        self.numeric_cols_ = [c for c in X.columns if c not in self.categorical_cols]
        # Which columns get the log is decided from TRAIN only, like everything else here.
        self.log_cols_ = self._find_skewed(X[self.numeric_cols_]) if self.log_skewed else []
        self.numeric_pipeline_ = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),  # learns train medians
                ("scale", StandardScaler()),  # learns train means and std devs
            ]
        ).fit(self._log(X[self.numeric_cols_]))

        # category -> ID, sorted so the IDs are the same on every machine.
        # IDs start at 1 because 0 is reserved for unseen categories.
        self.vocab_ = {
            col: {
                level: i + 1
                for i, level in enumerate(sorted(X[col].dropna().astype(str).unique()))
            }
            for col in self.categorical_cols
        }
        return self

    @property
    def cardinalities(self) -> list[int]:
        """Number of categories per text column seen in train (not counting the unknown ID)."""
        return [len(self.vocab_[col]) for col in self.categorical_cols]

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_num, x_cat): float32 of shape (rows, numeric cols) and int64 of shape
        (rows, text cols). Uses only what fit() learned from train."""
        x_num = self.numeric_pipeline_.transform(self._log(X[self.numeric_cols_])).astype(np.float32)
        if not self.categorical_cols:
            return x_num, np.zeros((len(X), 0), dtype=np.int64)
        x_cat = np.column_stack(
            [
                X[col].astype(str).map(self.vocab_[col]).fillna(UNKNOWN_ID).to_numpy(np.int64)
                for col in self.categorical_cols
            ]
        )
        return x_num, x_cat

    def fit_transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X).transform(X)


def embedding_dims(cardinalities, scale: float = 1.0, max_dim: int = 16) -> list[int]:
    """How many numbers each category's vector gets.

    Rule of thumb from the fast.ai library: about 1.6 * n ** 0.56 for a column with n
    categories (3 categories -> 3, 6 -> 4, 100 -> 21). `scale` multiplies it and is tuned;
    `max_dim` caps it. At least 1.
    """
    return [int(max(1, min(max_dim, round(scale * 1.6 * n**0.56)))) for n in cardinalities]


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
class EmbeddingMLP(nn.Module):
    """Embeddings for the text columns, joined to the numeric columns, then Dense layers.

    With no text columns (cardinalities=[]) it is a plain MLP, which is how the one-hot
    comparison model in the notebook is built: same code, only the input differs.
    """

    def __init__(
        self,
        n_numeric: int,
        cardinalities: list[int],
        emb_dims: list[int],
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        emb_dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()
        if len(cardinalities) != len(emb_dims):
            raise ValueError("need one embedding size per text column")

        # Kept so the model can be rebuilt from a saved file (see save_model / load_model).
        self.init_kwargs = {
            "n_numeric": int(n_numeric),
            "cardinalities": [int(n) for n in cardinalities],
            "emb_dims": [int(d) for d in emb_dims],
            "hidden": [int(h) for h in hidden],
            "dropout": float(dropout),
            "emb_dropout": float(emb_dropout),
            "batch_norm": bool(batch_norm),
        }

        # One table per text column: (categories + 1 for unknown) rows x d numbers.
        # padding_idx keeps the unknown row at zeros and never updates it.
        self.embeddings = nn.ModuleList(
            nn.Embedding(n + 1, d, padding_idx=UNKNOWN_ID) for n, d in zip(cardinalities, emb_dims)
        )
        self.emb_dropout = nn.Dropout(emb_dropout)

        layers = []
        width = n_numeric + sum(emb_dims)  # size of the joined input
        for units in hidden:
            layers.append(nn.Linear(width, units))
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            width = units
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)  # one output per customer

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Return one logit per customer. torch.sigmoid(logit) is the probability of buying.

        The sigmoid is not applied here because BCEWithLogitsLoss applies it inside the loss
        in a numerically safer way. predict_proba() applies it for real predictions.
        """
        parts = [x_num]
        if len(self.embeddings) > 0:
            vectors = [table(x_cat[:, i]) for i, table in enumerate(self.embeddings)]
            parts.append(self.emb_dropout(torch.cat(vectors, dim=1)))
        return self.head(self.body(torch.cat(parts, dim=1))).squeeze(1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Every setting that can be tuned. The defaults are a reasonable first try."""

    hidden: tuple[int, ...] = (128, 64)  # units per Dense layer
    dropout: float = 0.2  # after each Dense layer
    emb_dropout: float = 0.1  # on the joined embedding vectors
    emb_scale: float = 1.0  # multiplies the rule-of-thumb embedding size
    max_emb_dim: int = 16  # cap on any embedding size
    batch_norm: bool = True
    lr: float = 1e-3  # AdamW learning rate
    weight_decay: float = 1e-5  # AdamW weight decay (L2-style penalty)
    batch_size: int = 512
    pos_weight: float = 1.0  # loss weight on buyers; non-buyers / buyers in train is "balanced"
    max_epochs: int = 100
    patience: int = 10  # stop after this many epochs without a better validation PR-AUC
    lr_patience: int = 3  # halve the learning rate after this many epochs without a better
    #                       validation PR-AUC (0 = keep it fixed). Smaller than `patience`, so
    #                       the network gets a slower, finer try before early stopping gives up.

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = "-".join(str(h) for h in self.hidden)  # readable in a results table
        return d


def set_seed(seed: int) -> None:
    """Fix every random number generator the training uses."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensors(inputs):
    x_num, x_cat = inputs
    return torch.as_tensor(x_num, dtype=torch.float32), torch.as_tensor(x_cat, dtype=torch.int64)


@torch.no_grad()
def predict_proba(model: EmbeddingMLP, inputs, batch_size: int = 8192) -> np.ndarray:
    """Probability of buying for each row of `inputs` = (x_num, x_cat)."""
    model.eval()  # dropout off, batch norm uses its stored statistics
    x_num, x_cat = _to_tensors(inputs)
    out = [
        torch.sigmoid(model(x_num[i : i + batch_size], x_cat[i : i + batch_size]))
        for i in range(0, len(x_num), batch_size)
    ]
    return torch.cat(out).numpy()


def train_model(
    train_inputs,
    y_train,
    val_inputs,
    y_val,
    cardinalities: list[int],
    config: TrainConfig,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[EmbeddingMLP, pd.DataFrame]:
    """Train one network, with early stopping on validation PR-AUC.

    train_inputs, val_inputs: (x_num, x_cat) from TabularEncoder.transform
    y_train, y_val:           0/1 labels

    Returns (model, history). The model has the weights of its BEST validation epoch, not the
    last one. history has one row per epoch: train loss, validation loss and validation PR-AUC.
    """
    set_seed(seed)
    x_num, x_cat = _to_tensors(train_inputs)
    # np.array copies: under pandas 3 the labels from split_xy are read-only, which torch warns about.
    y = torch.as_tensor(np.array(y_train, dtype=np.float32))
    y_val = np.asarray(y_val)

    model = EmbeddingMLP(
        n_numeric=x_num.shape[1],
        cardinalities=cardinalities,
        emb_dims=embedding_dims(cardinalities, config.emb_scale, config.max_emb_dim),
        hidden=config.hidden,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
        batch_norm=config.batch_norm,
    )
    # Binary cross-entropy. pos_weight > 1 makes each missed buyer cost more, which helps
    # with only 14.6% buyers. It also pushes probabilities up, so they are scores for
    # ranking, not calibrated probabilities (calibration is a Progress Review 2 task).
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(config.pos_weight)))
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # mode="max" because a HIGHER validation PR-AUC is better.
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", factor=0.5, patience=config.lr_patience
        )
        if config.lr_patience > 0
        else None
    )
    shuffler = torch.Generator().manual_seed(seed)  # same batch order on every run

    val_num, val_cat = _to_tensors(val_inputs)
    val_y = torch.as_tensor(np.array(y_val, dtype=np.float32))

    best_pr_auc, best_epoch, best_state = -np.inf, 0, None
    history = []
    n = len(y)
    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter()

        # --- one pass over the train split, in shuffled mini-batches ---
        model.train()
        order = torch.randperm(n, generator=shuffler)
        total_loss = 0.0
        for start in range(0, n, config.batch_size):
            idx = order[start : start + config.batch_size]
            if len(idx) < 2:  # batch norm cannot work on a single row
                continue
            optimiser.zero_grad()
            loss = loss_fn(model(x_num[idx], x_cat[idx]), y[idx])
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(idx)

        # --- check on validation ---
        model.eval()
        with torch.no_grad():
            val_logits = model(val_num, val_cat)
            val_loss = loss_fn(val_logits, val_y).item()
        if not np.isfinite(val_loss):
            # The weights blew up (usually a learning rate that is too high). Stop here and
            # keep the best epoch so far, instead of crashing the whole random search.
            break
        val_pr_auc = average_precision_score(y_val, torch.sigmoid(val_logits).numpy())

        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / n,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
                "lr": optimiser.param_groups[0]["lr"],  # the rate used for this epoch
                "seconds": time.perf_counter() - started,
            }
        )
        if scheduler is not None:
            scheduler.step(val_pr_auc)
        if verbose:
            print(f"epoch {epoch:3d}  train loss {total_loss / n:.4f}  "
                  f"val loss {val_loss:.4f}  val PR-AUC {val_pr_auc:.4f}")

        # --- early stopping: keep the best weights, stop when they stop improving ---
        if val_pr_auc > best_pr_auc + 1e-4:
            best_pr_auc, best_epoch = val_pr_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_pr_auc = float("nan")  # diverged in the first epoch: the trial has no valid score
    model.eval()
    history = pd.DataFrame(history)
    history.attrs["best_epoch"] = best_epoch
    history.attrs["best_val_pr_auc"] = best_pr_auc
    return model, history


def sample_config(rng: np.random.Generator, space: dict, **fixed) -> TrainConfig:
    """Random search: pick one value per setting from `space` (a dict of lists).

    Settings in `fixed` are used as given. Anything in neither keeps its TrainConfig default.
    """
    choices = {}
    for name, values in space.items():
        choices[name] = values[rng.integers(len(values))]
    choices.update(fixed)
    return TrainConfig(**choices)


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------
def save_model(model: EmbeddingMLP, path) -> None:
    """Save the architecture and the weights. Save the TabularEncoder separately (joblib)."""
    torch.save({"init_kwargs": model.init_kwargs, "state_dict": model.state_dict()}, path)


def load_model(path) -> EmbeddingMLP:
    checkpoint = torch.load(path, weights_only=True)
    kwargs = dict(checkpoint["init_kwargs"])
    kwargs["hidden"] = tuple(kwargs["hidden"])
    model = EmbeddingMLP(**kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


__all__ = [
    "UNKNOWN_ID",
    "TabularEncoder",
    "embedding_dims",
    "EmbeddingMLP",
    "TrainConfig",
    "set_seed",
    "predict_proba",
    "train_model",
    "sample_config",
    "save_model",
    "load_model",
]
