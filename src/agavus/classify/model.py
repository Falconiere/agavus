"""Multinomial logistic regression: softmax + cross-entropy, full-batch GD.

Convex, deterministic, CPU-friendly. Sparse rows (dict index → weight);
weights are inspectable — the auditability payoff (docs/math/logistic-regression.md).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def softmax(logits: FloatArray) -> FloatArray:
  """Numerically stable row-wise softmax."""
  shifted = logits - logits.max(axis=1, keepdims=True)
  exp = np.exp(shifted)
  return exp / exp.sum(axis=1, keepdims=True)


def _logits(rows: Sequence[dict[int, float]], weights: FloatArray, bias: FloatArray) -> FloatArray:
  out = np.tile(bias, (len(rows), 1))
  for row_index, row in enumerate(rows):
    for feature, value in row.items():
      out[row_index] += value * weights[feature]
  return out


@dataclass(frozen=True)
class LogRegModel:
  """Trained linear classifier; every weight inspectable."""

  classes: tuple[str, ...]
  weights: FloatArray
  bias: FloatArray

  def logits(self, rows: Sequence[dict[int, float]]) -> FloatArray:
    """Raw scores per class."""
    return _logits(rows, self.weights, self.bias)

  def predict_proba(self, rows: Sequence[dict[int, float]]) -> FloatArray:
    """Class probabilities (rows sum to 1)."""
    return softmax(self.logits(rows))


@dataclass(frozen=True)
class TrainConfig:
  """Hyperparameters for the convex solver."""

  lr: float = 0.5
  epochs: int = 300
  l2: float = 1e-4


def train_logreg(
  rows: Sequence[dict[int, float]],
  labels: Sequence[str],
  classes: Sequence[str],
  *,
  n_features: int,
  config: TrainConfig | None = None,
) -> LogRegModel:
  """Minimize cross-entropy + L2 by deterministic gradient descent."""
  cfg = config if config is not None else TrainConfig()
  class_list = tuple(classes)
  targets = np.array([class_list.index(label) for label in labels])
  one_hot = np.eye(len(class_list))[targets]
  weights = np.zeros((n_features, len(class_list)))
  bias = np.zeros(len(class_list))
  count = len(rows)
  for _ in range(cfg.epochs):
    diff = (softmax(_logits(rows, weights, bias)) - one_hot) / count
    grad_w = np.zeros_like(weights)
    for row_index, row in enumerate(rows):
      for feature, value in row.items():
        grad_w[feature] += value * diff[row_index]
    weights -= cfg.lr * (grad_w + cfg.l2 * weights)
    bias -= cfg.lr * diff.sum(axis=0)
  return LogRegModel(classes=class_list, weights=weights, bias=bias)
