"""The reserve must win over training, even when no common epoch fits."""

import pytest

from agavus.classify.learned.budget import choose_epochs

EXPECTED_MAX_EPOCHS = 3


def test_common_epochs_reserve_evaluation_before_training() -> None:
  assert choose_epochs(remaining=100.0, epoch_seconds=10.0, evaluation_seconds=35.0) == 1
  assert choose_epochs(remaining=100.0, epoch_seconds=20.0, evaluation_seconds=35.0) == 0
  assert (
    choose_epochs(remaining=1000.0, epoch_seconds=10.0, evaluation_seconds=35.0)
    == EXPECTED_MAX_EPOCHS
  )
  with pytest.raises(ValueError, match="positive"):
    choose_epochs(remaining=100.0, epoch_seconds=0.0, evaluation_seconds=35.0)
