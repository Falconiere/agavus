"""Tests for the loss matrix (tests folder; safety ceiling enforced)."""

import pytest

from agavus.decide.losses import (
  REJECT_COSTS,
  SAFETY_CEILING,
  accept_action,
  build_matrix,
  validate_safety,
)


def test_matrix_covers_all_actions_and_states() -> None:
  matrix = build_matrix()
  assert set(matrix) == {"accept-F", "accept-M", "reject", "defer"}
  for row in matrix.values():
    assert set(row) == {"F", "M", "O"}


def test_safety_ceiling_values() -> None:
  matrix = build_matrix()
  assert matrix[accept_action("F")]["M"] == SAFETY_CEILING
  assert matrix["reject"]["M"] == REJECT_COSTS["M"]
  validate_safety(matrix)


def test_defer_is_uniform_safe_harbor() -> None:
  assert set(build_matrix()["defer"].values()) == {1.0}


def test_validation_rejects_excessive_cost() -> None:
  matrix = build_matrix()
  matrix["accept-F"]["M"] = SAFETY_CEILING + 1
  with pytest.raises(ValueError):
    validate_safety(matrix)


def test_validation_rejects_weakened_m_protection() -> None:
  matrix = build_matrix()
  matrix["reject"]["M"] = 2.0
  with pytest.raises(ValueError):
    validate_safety(matrix)
