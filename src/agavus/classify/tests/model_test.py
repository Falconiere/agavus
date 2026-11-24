"""Tests for the logistic regression core (tests folder, synthetic data)."""

import numpy as np

from agavus.classify.model import LogRegModel, TrainConfig, softmax, train_logreg


def _separable() -> tuple[list[dict[int, float]], list[str]]:
  rows: list[dict[int, float]] = []
  labels: list[str] = []
  for _ in range(6):
    rows.append({0: 1.0, 1: 0.1})
    labels.append("M")
    rows.append({1: 1.0, 0: 0.1})
    labels.append("F")
  return rows, labels


def test_softmax_rows_sum_to_one() -> None:
  probs = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
  assert np.allclose(probs.sum(axis=1), 1.0)


def _trained() -> LogRegModel:
  rows, labels = _separable()
  return train_logreg(rows, labels, ("F", "M"), n_features=2, config=TrainConfig(epochs=200))


def test_learns_separable_data() -> None:
  rows, labels = _separable()
  model = _trained()
  probs = model.predict_proba(rows)
  predictions = [model.classes[int(index)] for index in probs.argmax(axis=1)]
  assert predictions == labels


def test_deterministic_training() -> None:
  rows, labels = _separable()
  first = train_logreg(rows, labels, ("F", "M"), n_features=2, config=TrainConfig(epochs=50))
  second = train_logreg(rows, labels, ("F", "M"), n_features=2, config=TrainConfig(epochs=50))
  assert np.array_equal(first.weights, second.weights)


def test_weights_inspectable() -> None:
  model = _trained()
  m_index = model.classes.index("M")
  assert model.weights[0][m_index] > model.weights[1][m_index]
