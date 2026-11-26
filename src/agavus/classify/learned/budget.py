"""Conservative common-epoch scheduling with evaluation reserved first."""

import math

from agavus.classify.learned.types import SEEDS

MAX_EPOCHS = 3


def choose_epochs(*, remaining: float, epoch_seconds: float, evaluation_seconds: float) -> int:
  """All seeds and both tuning variants must fit; partial work has no winner."""
  if not all(math.isfinite(v) and v > 0 for v in (remaining, epoch_seconds, evaluation_seconds)):
    raise ValueError("timing estimates must be finite and positive")
  available = max(0.0, remaining - evaluation_seconds)
  return min(MAX_EPOCHS, int(available // (len(SEEDS) * 2 * epoch_seconds)))
