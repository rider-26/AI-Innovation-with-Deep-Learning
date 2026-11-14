"""Training loop for the MLP, with class weighting and early stopping."""

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shopsphere.config import SEED
from shopsphere.model import MLP, MLPParams

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_proba(model: MLP, X: np.ndarray) -> np.ndarray:
    """Return the predicted probability of buying for each row of X."""
    model.eval()  # switches dropout off, so predictions are deterministic
    with torch.no_grad():  # no gradients needed when only predicting
        logits = model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE))
    return torch.sigmoid(logits).cpu().numpy()


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: MLPParams,
    seed: int = SEED,
) -> tuple[MLP, pd.DataFrame]:
    """Train one MLP and return (the model at its best epoch, per-epoch history).

    The history has train and validation loss and PR-AUC for every epoch, which is what the
    overfitting check plots.
    """
    # Same seed -> same initial weights and same batch order -> same result.
    torch.manual_seed(seed)
    shuffle_rng = torch.Generator().manual_seed(seed)

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(np.array(y_train, dtype=np.float32))
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.as_tensor(np.array(y_val, dtype=np.float32), device=DEVICE)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=params.batch_size,
        shuffle=True,
        generator=shuffle_rng,
    )

    model = MLP(X_train.shape[1], params.hidden, params.dropout).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    # Class imbalance: weight each buyer by (non-buyers / buyers), about 5.85 on the real
    # data, so the rare class contributes as much to the loss as the common one.
    # Computed from the training labels only.
    n_pos = float(y_train_t.sum())
    pos_weight = torch.tensor((len(y_train_t) - n_pos) / n_pos, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best_score, best_state, epochs_without_improvement = -1.0, None, 0

    for epoch in range(1, params.max_epochs + 1):
        model.train()  # switches dropout on
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimiser.zero_grad()  # clear gradients left from the previous batch
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()  # compute gradients
            optimiser.step()  # update the weights

        # End of epoch: measure both splits with dropout off.
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_train_t.to(DEVICE)), y_train_t.to(DEVICE)).item()
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        train_pr_auc = average_precision_score(y_train, predict_proba(model, X_train))
        val_pr_auc = average_precision_score(y_val, predict_proba(model, X_val))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_pr_auc": train_pr_auc,
                "val_pr_auc": val_pr_auc,
            }
        )

        # Early stopping on validation PR-AUC: keep the best weights seen so far, and stop
        # once `patience` epochs pass without beating them. This limits overfitting and
        # means the number of epochs does not have to be tuned by hand.
        if val_pr_auc > best_score:
            best_score = val_pr_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= params.patience:
                break

    model.load_state_dict(best_state)  # roll back to the best epoch, not the last one
    return model, pd.DataFrame(history)
