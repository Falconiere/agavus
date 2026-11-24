"""Tests for classification metrics (tests folder, hand-computed cases)."""

from agavus.evals.metrics import accuracy, confusion, macro_f1


def test_accuracy() -> None:
  assert accuracy(["F", "M", "O"], ["F", "M", "F"]) == 2 / 3
  assert accuracy([], []) == 0.0


def test_confusion_counts() -> None:
  matrix = confusion(["F", "M", "M"], ["F", "F", "M"], ("F", "M"))
  assert matrix["F"]["F"] == 1
  assert matrix["F"]["M"] == 1
  assert matrix["M"]["M"] == 1


def test_macro_f1_perfect() -> None:
  assert macro_f1(["F", "M"], ["F", "M"], ("F", "M")) == 1.0


def test_macro_f1_partial() -> None:
  score = macro_f1(["F", "F", "M"], ["F", "M", "M"], ("F", "M"))
  assert 0.0 < score < 1.0
