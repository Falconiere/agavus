"""Shared split inspection without optional model imports."""

from agavus.classify.learned.partitions import Partitions


def partition_locations(parts: Partitions) -> dict[int, str]:
  """Map each frozen row index to its partition name."""
  return {index: name for name, rows in vars(parts).items() for index in rows}
