"""The deep learning model: a multilayer perceptron (MLP) for tabular data."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class MLPParams:
    """Every setting that tuning is allowed to change, in one place."""

    hidden: tuple[int, ...] = (128, 64)  # units in each hidden layer
    dropout: float = 0.3  # share of units switched off at random in training
    lr: float = 1e-3  # Adam learning rate
    weight_decay: float = 0.0  # L2 penalty on the weights
    batch_size: int = 512
    max_epochs: int = 100  # upper limit; early stopping normally ends sooner
    patience: int = 5  # epochs to wait for a better validation PR-AUC

    def to_dict(self) -> dict:
        return asdict(self)


class MLP(nn.Module):
    """Linear -> ReLU -> Dropout for each hidden layer, then one output unit.

    The output is a *logit* (any real number), not a probability. The sigmoid that turns it
    into a probability is applied inside the loss during training (BCEWithLogitsLoss, which
    is more numerically stable than a separate sigmoid) and in `train.predict_proba`.
    """

    def __init__(self, n_inputs: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_inputs
        for units in hidden:
            layers += [nn.Linear(width, units), nn.ReLU(), nn.Dropout(dropout)]
            width = units
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape (batch,) to match the labels
