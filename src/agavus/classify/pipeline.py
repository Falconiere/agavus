"""Training pipeline: fit-on-train-only, held-out calibration, metrics.

Deterministic split (sorted, every k-th to holdout). The vectorizer, model,
and temperature are fitted exclusively on the training side; the holdout
measures honest generalization (docs/evals.md).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from agavus.classify.calibrate import (
  apply_temperature,
  expected_calibration_error,
  fit_temperature,
)
from agavus.classify.features import TfidfVectorizer, VectorizerConfig
from agavus.classify.model import LogRegModel, TrainConfig, softmax, train_logreg
from agavus.classify.seed import SEED_QUERIES
from agavus.corpus.storage import CorpusStorage
from agavus.evals.metrics import accuracy, macro_f1

FloatArray = NDArray[np.float64]
HOLDOUT_EVERY = 5
MIN_HOLDOUT_FOR_CALIBRATION = 5


@dataclass(frozen=True)
class TrainReport:
  """Held-out quality summary — the artifact's birth certificate."""

  holdout_accuracy: float
  holdout_macro_f1: float
  holdout_ece: float
  temperature: float
  n_train: int
  n_holdout: int


@dataclass(frozen=True)
class TrainedClassifier:
  """Versioned bundle: vocabulary + weights + temperature.

  Persistence lives in classify/store.py (SQLite artifacts, no JSON files).
  """

  vectorizer: TfidfVectorizer
  model: LogRegModel
  temperature: float

  def predict_proba(self, texts: Sequence[str]) -> FloatArray:
    """Calibrated class probabilities for raw texts."""
    rows = self.vectorizer.transform(texts)
    return softmax(apply_temperature(self.model.logits(rows), self.temperature))

  def probabilities(self, text: str) -> dict[str, float]:
    """Calibrated {class: probability} for one text."""
    row = self.predict_proba([text])[0]
    return {label: float(value) for label, value in zip(self.model.classes, row, strict=True)}

  def top_features(self, text: str, k: int = 5) -> list[tuple[str, float]]:
    """Largest logit contributions to the winning class (auditability)."""
    row = self.vectorizer.transform([text])[0]
    winner = int(self.predict_proba([text])[0].argmax())
    inverse = {index: term for term, index in self.vectorizer.vocab.items()}
    contributions = [
      (inverse[index], value * float(self.model.weights[index][winner]))
      for index, value in row.items()
    ]
    contributions.sort(key=lambda pair: -abs(pair[1]))
    return contributions[:k]


def training_set(storage: CorpusStorage) -> tuple[list[str], list[str]]:
  """Labeled corpus texts + authored seeds (shared by CLI and Space)."""
  labels = {decision.record_id: decision for decision in storage.load_labels()}
  texts: list[str] = []
  targets: list[str] = []
  for document in storage.load_texts():
    decision = labels.get(document.id)
    if decision is not None and decision.domain_label is not None:
      texts.append(document.text)
      targets.append(decision.domain_label)
  texts += [text for text, _ in SEED_QUERIES]
  targets += [label for _, label in SEED_QUERIES]
  return texts, targets


def train_classifier(
  texts: Sequence[str],
  labels: Sequence[str],
  *,
  vectorizer_config: VectorizerConfig | None = None,
  train_config: TrainConfig | None = None,
) -> tuple[TrainedClassifier, TrainReport]:
  """Train + calibrate + report; raises on degenerate inputs."""
  classes = tuple(sorted(set(labels)))
  min_classes = 2
  if len(classes) < min_classes or len(texts) != len(labels):
    raise ValueError("need >= 2 classes and aligned texts/labels")
  order = sorted(range(len(texts)), key=lambda index: (labels[index], texts[index]))
  holdout = set(order[::HOLDOUT_EVERY])
  if not holdout or len(holdout) == len(texts):
    raise ValueError("degenerate holdout split")
  train_texts = [text for index, text in enumerate(texts) if index not in holdout]
  train_labels = [label for index, label in enumerate(labels) if index not in holdout]
  hold_texts = [text for index, text in enumerate(texts) if index in holdout]
  hold_labels = [label for index, label in enumerate(labels) if index in holdout]
  vectorizer = TfidfVectorizer(vectorizer_config)
  vectorizer.fit(train_texts)
  model = train_logreg(
    vectorizer.transform(train_texts),
    train_labels,
    classes,
    n_features=len(vectorizer.vocab),
    config=train_config,
  )
  logits = model.logits(vectorizer.transform(hold_texts))
  targets = [classes.index(label) for label in hold_labels]
  temperature = 1.0
  if len(targets) >= MIN_HOLDOUT_FOR_CALIBRATION:
    temperature = fit_temperature(logits, targets)
  probs = softmax(apply_temperature(logits, temperature))
  predictions = [classes[int(index)] for index in probs.argmax(axis=1)]
  report = TrainReport(
    holdout_accuracy=accuracy(predictions, hold_labels),
    holdout_macro_f1=macro_f1(predictions, hold_labels, classes),
    holdout_ece=expected_calibration_error(probs, targets),
    temperature=temperature,
    n_train=len(train_texts),
    n_holdout=len(hold_texts),
  )
  return TrainedClassifier(vectorizer, model, temperature), report
