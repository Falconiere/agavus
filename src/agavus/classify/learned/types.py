"""Immutable experiment examples and fixed task vocabulary."""

from dataclasses import dataclass
from datetime import date

RELATIONS = ("support", "contradiction", "unknown")
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Sample:
  """One provenance-bearing document or synthetic passage/statement pair."""

  id: str
  text: str
  label: str
  group: str
  origin_ref: str
  label_source: str
  published_at: date | None = None
  auxiliary: bool = False
  statement: str = ""
  category: str = ""
  template: str = ""

  @property
  def task(self) -> str:
    """Relation supervision never enters the area registry."""
    return "relation" if self.statement else "area"
