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
from agavus.classify.seed_scenarios import SCENARIOS
from agavus.retrieve.encoder import MODEL_VERSION, Encoder

ROUTER_VERSION = "router-v2"
TRAINING: tuple[tuple[str, str], ...] = (
  SEED_QUERIES
  + EVERYDAY_QUERIES
  + tuple((text, label) for _, label, questions in SCENARIOS for text in questions)
)
_SCENARIO_GROUPS = {text: group for group, _, questions in SCENARIOS for text in questions}
FOLDS = 5
MIN_GROUPS = 2
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
  content = repr((ROUTER_VERSION, MODEL_VERSION, TRAINING, SCENARIOS, CONFIG, SPREAD, FOLDS))
  return hashlib.sha256(content.encode()).hexdigest()


def _fit(vectors: FloatArray, labels: Sequence[str], classes: Sequence[str]) -> LogRegModel:
  rows = [{index: float(value) for index, value in enumerate(row)} for row in vectors]
  return train_logreg(rows, labels, classes, n_features=vectors.shape[1], config=CONFIG)


def calibration_folds(
  labels: Sequence[str], groups: Sequence[str], folds: int = FOLDS
) -> tuple[tuple[int, ...], ...]:
  """Stratified grouped folds: a scenario never helps predict its own variant."""
  if len(labels) != len(groups) or folds < MIN_GROUPS:
    raise ValueError("need aligned labels/groups and at least two folds")
  members: dict[str, list[int]] = {}
  for index, group in enumerate(groups):
    members.setdefault(group, []).append(index)
  if any(len({labels[index] for index in rows}) != 1 for rows in members.values()):
    raise ValueError("conflicting labels within a calibration group")
  held: list[list[int]] = [[] for _ in range(folds)]
  random = np.random.default_rng(0)
  for label in sorted(set(labels)):
    matching = sorted(group for group, rows in members.items() if labels[rows[0]] == label)
    if len(matching) < MIN_GROUPS:
      raise ValueError("need at least two independent groups per class")
    for offset, position in enumerate(random.permutation(len(matching))):
      held[offset % folds].extend(members[matching[int(position)]])
  return tuple(tuple(sorted(rows)) for rows in held if rows)


def _normalization(vectors: FloatArray) -> tuple[FloatArray, float]:
  center = vectors.mean(axis=0)
  spread = float(np.sqrt(((vectors - center) ** 2).sum(axis=1).mean()))
  if not np.isfinite(spread) or spread <= 0:
    raise ValueError("router training needs finite, distinct vectors")
  return center, SPREAD / spread


def _out_of_fold_logits(
  vectors: FloatArray, labels: Sequence[str], classes: Sequence[str], groups: Sequence[str]
) -> FloatArray:
  """Each score excludes its scenario from weights and normalization alike."""
  logits = np.zeros((len(labels), len(classes)))
  for picked in calibration_folds(labels, groups):
    held = set(picked)
    kept = [index for index in range(len(labels)) if index not in held]
    center, scale = _normalization(vectors[kept])
    normalized = (vectors - center) * scale
    model = _fit(normalized[kept], [labels[index] for index in kept], classes)
    logits[list(picked)] = normalized[list(picked)] @ model.weights + model.bias
  return logits


def fit_router(encoder: Encoder, training: Sequence[tuple[str, str]] = TRAINING) -> RouterWeights:
  """Fit on seeds; calibrate the temperature on out-of-fold scores."""
  labels = [label for _, label in training]
  groups = [_SCENARIO_GROUPS.get(text, " ".join(text.casefold().split())) for text, _ in training]
  classes = tuple(sorted(set(labels)))
  raw = encoder.encode_queries([text for text, _ in training])
  center, scale = _normalization(raw)
  vectors = (raw - center) * scale
  targets = [classes.index(label) for label in labels]
  temperature = fit_temperature(_out_of_fold_logits(raw, labels, classes, groups), targets)
  return RouterWeights(_fit(vectors, labels, classes), center, scale, temperature)
