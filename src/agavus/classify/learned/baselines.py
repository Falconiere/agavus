"""Unchanged TF-IDF features and a frozen raw-token E5 linear comparison."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from agavus.classify.features import TfidfVectorizer
from agavus.classify.learned.dataset import Dataset
from agavus.classify.learned.engine import task_classes
from agavus.classify.learned.state import ExperimentState
from agavus.classify.learned.train import TokenizedSample, check_deadline
from agavus.classify.model import FloatArray, LogRegModel, train_logreg
from agavus.classify.pipeline import TrainedClassifier
from agavus.classify.router import _normalization
from agavus.classify.store import save_model


class Predictor(Protocol):
  """The comparison needs scores only; it cannot activate application routing."""

  def logits(self, examples: Sequence[TokenizedSample]) -> FloatArray:
    """Uncalibrated scores in task_classes order, from one task at a time."""
    ...


def _text(item: TokenizedSample) -> str:
  return (
    f"{item.sample.text}\n{item.sample.statement}" if item.sample.statement else item.sample.text
  )


@dataclass(frozen=True)
class TfidfBaseline:
  """Vocabulary, IDF and weights learned exclusively from the shared training rows."""

  models: dict[str, TrainedClassifier]

  def logits(self, examples: Sequence[TokenizedSample]) -> FloatArray:
    """Retain every production feature family in this baseline."""
    model = self.models[examples[0].sample.task]
    return model.model.logits(model.vectorizer.transform([_text(item) for item in examples]))

  def save_calibrated(self, folder: Path, temperatures: tuple[float, float]) -> None:
    """Persist the calibration used in the comparison, rather than initial raw scores."""
    for task, temperature in zip(("area", "relation"), temperatures, strict=True):
      fitted = replace(self.models[task], temperature=temperature)
      save_model(folder / "baselines.db", f"tfidf:{task}", fitted)


def fit_tfidf(
  training: Sequence[TokenizedSample],
  state: ExperimentState,
  deadline: float,
) -> TfidfBaseline:
  """Fit both tasks independently; no held-out text enters the vocabulary."""
  models: dict[str, TrainedClassifier] = {}
  for task in ("area", "relation"):
    check_deadline(deadline)
    items = [item for item in training if item.sample.task == task]
    texts = [_text(item) for item in items]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    rows = vectorizer.transform(texts)
    model = train_logreg(
      rows, [i.sample.label for i in items], task_classes(task), n_features=len(vectorizer.vocab)
    )
    models[task] = TrainedClassifier(vectorizer, model, 1.0)
    save_model(state.folder / "baselines.db", f"tfidf:{task}", models[task])
  return TfidfBaseline(models)


@dataclass(frozen=True)
class FrozenHead:
  """Training-only centering, scale and balanced linear weights."""

  model: LogRegModel
  center: FloatArray
  scale: float


@dataclass(frozen=True)
class FrozenBaseline:
  """Precomputed vectors are observations, never fitted on final labels."""

  vectors: dict[str, FloatArray]
  heads: dict[str, FrozenHead]

  def logits(self, examples: Sequence[TokenizedSample]) -> FloatArray:
    """Document-averaged features equal averaged linear window logits."""
    head = self.heads[examples[0].sample.task]
    matrix = (np.stack([self.vectors[i.sample.id] for i in examples]) - head.center) * head.scale
    return np.asarray(matrix @ head.model.weights + head.model.bias, dtype=np.float64)


def fit_frozen(vectors: FloatArray, dataset: Dataset, state: ExperimentState) -> FrozenBaseline:
  """No trainable encoder updates in this baseline, with the same training membership."""
  heads: dict[str, FrozenHead] = {}
  for task in ("area", "relation"):
    indexes = [i for i in dataset.partitions.train if dataset.samples[i].task == task]
    training = vectors[indexes]
    center, scale = _normalization(training)
    normalized = (training - center) * scale
    rows = [dict(enumerate(map(float, row))) for row in normalized]
    model = train_logreg(
      rows,
      [dataset.samples[i].label for i in indexes],
      task_classes(task),
      n_features=vectors.shape[1],
    )
    heads[task] = FrozenHead(model, center, scale)
    state.put(
      f"frozen:{task}",
      {
        "center": center.tolist(),
        "scale": scale,
        "weights": model.weights.tolist(),
        "bias": model.bias.tolist(),
        "classes": model.classes,
      },
    )
  return FrozenBaseline(dict(zip((s.id for s in dataset.samples), vectors, strict=True)), heads)
