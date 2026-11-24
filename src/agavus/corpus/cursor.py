"""Resume cursor for a newest-to-oldest walk. Legacy date watermarks are ignored."""

import json
from dataclasses import dataclass

from agavus.corpus.jsonutil import as_dict, as_list, as_str, parse_date

_MONTH_LEN = 7
_YEAR_LEN = 4


@dataclass(frozen=True)
class WalkState:
  """search_after cursor, newest hit, month being walked, and whether the tail is done."""

  after: tuple[str, ...] | None
  newest: tuple[str, ...] | None
  done: bool
  month: str | None = None
  empty_months: int = 0


def encode(state: WalkState) -> str:
  """Stable JSON watermark. Sort values are echoed verbatim to search_after."""
  payload: dict[str, object] = {
    "after": list(state.after) if state.after else None,
    "done": state.done,
    "empty_months": state.empty_months,
    "month": state.month,
    "newest": list(state.newest) if state.newest else None,
  }
  return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _pair(value: object) -> tuple[str, ...] | None:
  items = tuple(as_str(item) for item in as_list(value))
  if not items or any(not item for item in items):
    return None
  return items


def _month(value: object) -> str | None:
  text = as_str(value)
  year = text[:_YEAR_LEN]
  if len(text) == _MONTH_LEN and text[_YEAR_LEN] == "-" and year.isdigit() and text[5:].isdigit():
    return text
  return None


def _count(value: object) -> int:
  if isinstance(value, int) and value >= 0:
    return value
  return 0


def decode(raw: str | None) -> WalkState:
  """Load a cursor. A bare ISO date (the old watermark) is discarded."""
  if not raw or not raw.startswith("{"):
    return WalkState(None, None, False)
  try:
    payload = json.loads(raw)
  except json.JSONDecodeError:
    return WalkState(None, None, False)
  body = as_dict(payload)
  done = body.get("done") is True
  return WalkState(
    _pair(body.get("after")),
    _pair(body.get("newest")),
    done,
    _month(body.get("month")),
    _count(body.get("empty_months")),
  )


def comes_before(candidate: tuple[str, ...], frontier: tuple[str, ...]) -> bool:
  """True when candidate is strictly newer under a descending date sort."""
  for left, right in zip(candidate, frontier, strict=False):
    if left == right:
      continue
    left_date = parse_date(left)
    right_date = parse_date(right)
    if left_date is not None and right_date is not None and left_date != right_date:
      return left_date > right_date
    return left > right
  return False
