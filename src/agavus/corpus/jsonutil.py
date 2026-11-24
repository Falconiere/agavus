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


_COMPACT_DATE = 8
_ISO_DATE = 10


def parse_date(value: object) -> date | None:
  """Parse an ISO date or compact YYYYMMDD[HHMMSS]; None when absent or invalid.

  DataJud stores `dataAjuizamento` as 14 digits (`YYYYMMDDHHmmss`) and, on
  some rows, as ISO. The first eight digits are the Brasília calendar date.
  """
  text = as_str(value)
  if len(text) >= _COMPACT_DATE and text[:_COMPACT_DATE].isdigit():
    try:
      return date(int(text[0:4]), int(text[4:6]), int(text[6:_COMPACT_DATE]))
    except ValueError:
      return None
  if len(text) < _ISO_DATE:
    return None
  try:
    return date.fromisoformat(text[:_ISO_DATE])
  except ValueError:
    return None
