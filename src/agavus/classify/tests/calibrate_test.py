"""Tests for calibration (tests folder, synthetic logits)."""

import numpy as np

from agavus.classify.calibrate import (
  apply_temperature,
  expected_calibration_error,
  fit_temperature,
)
from agavus.classify.model import softmax


def test_temperature_reduces_overconfidence() -> None:
  logits = np.array([[8.0, 0.0]] * 8 + [[0.0, 8.0]] * 2)
  targets = [0] * 5 + [1] * 5  # 50% accuracy despite extreme logits
  temperature = fit_temperature(logits, targets)
  assert temperature > 1.0
  before = expected_calibration_error(softmax(logits), targets)
  after = expected_calibration_error(softmax(apply_temperature(logits, temperature)), targets)
  assert after < before


def test_temperature_preserves_argmax() -> None:
  logits = np.array([[3.0, 1.0], [0.5, 2.0]])
  scaled = apply_temperature(logits, 2.5)
  assert np.array_equal(logits.argmax(axis=1), scaled.argmax(axis=1))


def test_ece_perfect_calibration() -> None:
  probs = np.array([[0.75, 0.25]] * 4 + [[0.25, 0.75]] * 4)
  targets = [0, 0, 0, 1, 1, 1, 1, 0]
  tolerance = 1e-9
  assert expected_calibration_error(probs, targets) < tolerance
