"""Result of one ingest invocation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestBatch:
  """How far a newest-to-oldest walk got."""

  new: int
  done: bool
  cursor: str
  unknown: tuple[str, ...] = ()
