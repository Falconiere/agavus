"""Shared synthetic router for tests across domains."""

from collections.abc import Sequence


class UnsureRouter:
  """Gives every question the same undecided probabilities."""

  def probabilities(self, questions: Sequence[str]) -> list[dict[str, float]]:
    """One undecided mapping per question."""
    return [{"F": 0.4, "M": 0.4, "O": 0.2} for _ in questions]
