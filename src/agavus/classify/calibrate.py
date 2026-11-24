"""Calibration: temperature scaling (multiclass Platt) + ECE.

Fit on held-out logits only; the decision layer spends real costs against
these probabilities (docs/math/calibration.md).
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from agavus.classify.model import softmax

FloatArray = NDArray[np.float64]
TEMPERATURE_GRID = tuple(float(value) for value in np.linspace(0.2, 5.0, 49))


def fit_temperature(logits: FloatArray, targets: Sequence[int]) -> float:
  """Grid-search T minimizing held-out NLL (deterministic)."""
  best_t = 1.0
  best_nll = float("inf")
  for temperature in TEMPERATURE_GRID:
    probs = softmax(logits / temperature)
    nll = -float(np.log(probs[np.arange(len(targets)), targets] + 1e-12).mean())
    if nll < best_nll:
      best_nll = nll
      best_t = temperature
  return best_t


def apply_temperature(logits: FloatArray, temperature: float) -> FloatArray:
  """Scale logits; argmax unchanged, confidence honest."""
  return logits / temperature


def expected_calibration_error(
  probs: FloatArray, targets: Sequence[int], n_bins: int = 10
) -> float:
  """Mean |confidence - accuracy| over confidence bins, weighted by bin size."""
  confidences = probs.max(axis=1)
  predictions = probs.argmax(axis=1)
  correct = (predictions == np.array(targets)).astype(float)
  error = 0.0
  for bin_index in range(n_bins):
    low = bin_index / n_bins
    high = (bin_index + 1) / n_bins
    mask = (confidences > low) & (confidences <= high)
    if bin_index == 0:
      mask = (confidences >= low) & (confidences <= high)
    size = int(mask.sum())
    if size:
      error += (size / len(targets)) * abs(
        float(correct[mask].mean()) - float(confidences[mask].mean())
      )
  return error
