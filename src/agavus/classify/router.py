"""Question router: logistic regression over sentence vectors.

Questions are short and colloquial, so they get their own model, trained only
on authored seeds; documents and the no-weights fallback keep TF-IDF. The
decision rule and loss matrix are the same for both.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from agavus.classify.calibrate import fit_temperature
from agavus.classify.model import FloatArray, LogRegModel, TrainConfig, softmax, train_logreg
from agavus.classify.seed import SEED_QUERIES
from agavus.classify.seed_everyday import EVERYDAY_QUERIES
from agavus.retrieve.encoder import MODEL_VERSION, Encoder

ROUTER_VERSION = "router-v1"
TRAINING: tuple[tuple[str, str], ...] = SEED_QUERIES + EVERYDAY_QUERIES
FOLDS = 5
# Sentence vectors are tiny numbers around a shared center; spread them out.
SPREAD = 4.0
CONFIG = TrainConfig(lr=1.0, epochs=1500, l2=1e-3)


@dataclass(frozen=True)
class RouterWeights:
  """Everything needed to route, apart from the encoder."""

  model: LogRegModel
  center: FloatArray
  scale: float
  temperature: float


@dataclass(frozen=True)
class QueryRouter:
  """Calibrated area probabilities for questions."""

  encoder: Encoder
  weights: RouterWeights

  def probabilities(self, questions: Sequence[str]) -> list[dict[str, float]]:
    """One {area: probability} per question."""
    fitted = self.weights
    vectors = (self.encoder.encode_queries(questions) - fitted.center) * fitted.scale
    logits = vectors @ fitted.model.weights + fitted.model.bias
    return [
      {label: float(value) for label, value in zip(fitted.model.classes, row, strict=True)}
      for row in softmax(logits / fitted.temperature)
    ]


def fingerprint() -> str:
  """Identity of encoder, training data and settings; changes force a refit."""
  content = repr((ROUTER_VERSION, MODEL_VERSION, TRAINING, CONFIG, SPREAD, FOLDS))
  return hashlib.sha256(content.encode()).hexdigest()


def _fit(vectors: FloatArray, labels: Sequence[str], classes: Sequence[str]) -> LogRegModel:
  rows = [{index: float(value) for index, value in enumerate(row)} for row in vectors]
  return train_logreg(rows, labels, classes, n_features=vectors.shape[1], config=CONFIG)


def _out_of_fold_logits(
  vectors: FloatArray, labels: Sequence[str], classes: Sequence[str]
) -> FloatArray:
  """Scores for each example from a model that never saw it."""
  order = np.random.default_rng(0).permutation(len(labels))
  logits = np.zeros((len(labels), len(classes)))
  for fold in range(FOLDS):
    held = set(order[fold::FOLDS].tolist())
    kept = [index for index in range(len(labels)) if index not in held]
    model = _fit(vectors[kept], [labels[index] for index in kept], classes)
    picked = sorted(held)
    logits[picked] = vectors[picked] @ model.weights + model.bias
  return logits


def fit_router(encoder: Encoder, training: Sequence[tuple[str, str]] = TRAINING) -> RouterWeights:
  """Fit on seeds; calibrate the temperature on out-of-fold scores."""
  labels = [label for _, label in training]
  classes = tuple(sorted(set(labels)))
  raw = encoder.encode_queries([text for text, _ in training])
  center = raw.mean(axis=0)
  centered = raw - center
  scale = SPREAD / float(np.sqrt((centered**2).sum(axis=1).mean()))
  vectors = centered * scale
  targets = [classes.index(label) for label in labels]
  temperature = fit_temperature(_out_of_fold_logits(vectors, labels, classes), targets)
  return RouterWeights(_fit(vectors, labels, classes), center, scale, temperature)
