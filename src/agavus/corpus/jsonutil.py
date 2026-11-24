"""Typed helpers for navigating parsed JSON without Any."""

from datetime import date
from typing import cast


def as_dict(value: object) -> dict[str, object]:
  """Return value as a JSON object dict, or empty when not a dict."""
  return cast(dict[str, object], value) if isinstance(value, dict) else {}


def as_list(value: object) -> list[object]:
  """Return value as a JSON array list, or empty when not a list."""
  return cast(list[object], value) if isinstance(value, list) else []


def as_str(value: object) -> str:
  """Return value as a string, or empty when not a string."""
  return value if isinstance(value, str) else ""


def as_int(value: object) -> int:
  """Return value as an int, or 0 when not numeric."""
  return value if isinstance(value, int) else 0


def parse_date(value: object) -> date | None:
  """Parse an ISO date or datetime string; None when absent/invalid."""
  text = as_str(value)
  if not text:
    return None
  try:
    return date.fromisoformat(text[:10])
  except ValueError:
    return None
