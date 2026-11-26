"""Case-grouped training, separate calibration, and final holdout metrics."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

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
from agavus.classify.splits import TrainingData, split_examples
from agavus.corpus.records import TextSource
from agavus.corpus.storage import CorpusStorage
from agavus.evals.metrics import accuracy, macro_f1

FloatArray = NDArray[np.float64]
_CNJ = re.compile(r"(?<!\d)\d{7}-?\d{2}\.?\d{4}\.?\d\.?\d{2}\.?\d{4}(?!\d)")


@dataclass(frozen=True)
class TrainReport:
  """Held-out quality summary — the artifact's birth certificate."""

  holdout_accuracy: float
  holdout_macro_f1: float
  holdout_ece: float
  temperature: float
  n_train: int
  n_holdout: int
  n_calibration: int = 0


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


def training_set(storage: CorpusStorage) -> TrainingData:
  """Resolved corpus labels + seeds, retaining case identity and dates."""
  labels = {
    decision.record_id: decision
    for decision in storage.load_labels()
    if decision.domain_label is not None and not decision.needs_review
  }
  texts: list[str] = []
  targets: list[str] = []
  groups: list[str] = []
  dates: list[date | None] = []
  for document in storage.load_texts_by_ids(tuple(labels)).values():
    match = _CNJ.search(document.origin_ref)
    if document.source != TextSource.STATUTE and match is None:
      continue
    decision = labels[document.id]
    if decision.domain_label is not None:
      texts.append(document.text)
      targets.append(decision.domain_label)
      groups.append(re.sub(r"\D", "", match[0]) if match else document.origin_ref)
      dates.append(document.published_at)
  texts += [text for text, _ in SEED_QUERIES]
  targets += [label for _, label in SEED_QUERIES]
  groups += [text for text, _ in SEED_QUERIES]
  dates += [None] * len(SEED_QUERIES)
  return TrainingData(tuple(texts), tuple(targets), tuple(groups), tuple(dates))


def train_classifier(
  texts: Sequence[str] | TrainingData,
  labels: Sequence[str] = (),
  *,
  vectorizer_config: VectorizerConfig | None = None,
  train_config: TrainConfig | None = None,
) -> tuple[TrainedClassifier, TrainReport]:
  """Train + calibrate + report; raises on degenerate inputs."""
  if isinstance(texts, TrainingData):
    if labels:
      raise ValueError("labels are already included in TrainingData")
    data = texts
    texts, labels = data.texts, data.labels
  else:
    data = TrainingData(tuple(texts), tuple(labels), tuple(texts), (None,) * len(texts))
  classes = tuple(sorted(set(labels)))
  min_classes = 2
  if len(classes) < min_classes or len(texts) != len(labels):
    raise ValueError("need >= 2 classes and aligned texts/labels")
  split = split_examples(texts, labels, groups=data.groups, dates=data.dates)
  train_texts = [texts[index] for index in split.train]
  train_labels = [labels[index] for index in split.train]
  hold_texts = [texts[index] for index in split.holdout]
  hold_labels = [labels[index] for index in split.holdout]
  vectorizer = TfidfVectorizer(vectorizer_config)
  vectorizer.fit(train_texts)
  model = train_logreg(
    vectorizer.transform(train_texts),
    train_labels,
    classes,
    n_features=len(vectorizer.vocab),
    config=train_config,
  )
  calibration = model.logits(vectorizer.transform([texts[index] for index in split.calibration]))
  temperature = fit_temperature(calibration, [classes.index(labels[i]) for i in split.calibration])
  logits = model.logits(vectorizer.transform(hold_texts))
  targets = [classes.index(label) for label in hold_labels]
  probs = softmax(apply_temperature(logits, temperature))
  predictions = [classes[int(index)] for index in probs.argmax(axis=1)]
  report = TrainReport(
    holdout_accuracy=accuracy(predictions, hold_labels),
    holdout_macro_f1=macro_f1(predictions, hold_labels, classes),
    holdout_ece=expected_calibration_error(probs, targets),
    temperature=temperature,
    n_train=len(train_texts),
    n_holdout=len(hold_texts),
    n_calibration=len(split.calibration),
  )
  return TrainedClassifier(vectorizer, model, temperature), report
