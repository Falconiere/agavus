"""Training pipeline: fit-on-train-only, held-out calibration, metrics.

Deterministic split (sorted, every k-th to holdout). The vectorizer, model,
and temperature are fitted exclusively on the training side; the holdout
measures honest generalization (docs/evals.md).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from agavus.classify.calibrate import (
  apply_temperature,
  expected_calibration_error,
  fit_temperature,
)
from agavus.classify.features import TfidfVectorizer, VectorizerConfig
from agavus.classify.model import LogRegModel, TrainConfig, softmax, train_logreg
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
  """Versioned bundle: vocabulary + weights + temperature."""

  vectorizer: TfidfVectorizer
  model: LogRegModel
  temperature: float

  def predict_proba(self, texts: Sequence[str]) -> FloatArray:
    """Calibrated class probabilities for raw texts."""
    rows = self.vectorizer.transform(texts)
    return softmax(apply_temperature(self.model.logits(rows), self.temperature))

  def save(self, folder: Path) -> None:
    """Persist the bundle (outside git; rebuildable by retraining)."""
    folder.mkdir(parents=True, exist_ok=True)
    config = self.vectorizer.config
    vocab = {
      "vocab": self.vectorizer.vocab,
      "idf": self.vectorizer.idf,
      "min_df": config.min_df,
      "max_features": config.max_features,
      "char_sizes": list(config.char_sizes),
    }
    (folder / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    np.savez(folder / "model.npz", weights=self.model.weights, bias=self.model.bias)
    meta = {"classes": list(self.model.classes), "temperature": self.temperature}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

  @staticmethod
  def load(folder: Path) -> "TrainedClassifier":
    """Reload a persisted bundle."""
    vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    config = VectorizerConfig(
      min_df=vocab["min_df"],
      max_features=vocab["max_features"],
      char_sizes=tuple(vocab["char_sizes"]),
    )
    vectorizer = TfidfVectorizer(config)
    vectorizer.vocab = vocab["vocab"]
    vectorizer.idf = vocab["idf"]
    with np.load(folder / "model.npz") as arrays:
      model = LogRegModel(
        classes=tuple(meta["classes"]), weights=arrays["weights"], bias=arrays["bias"]
      )
    return TrainedClassifier(vectorizer, model, float(meta["temperature"]))


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
