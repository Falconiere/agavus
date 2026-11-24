"""TPU code → area mapping: curated, versioned (docs/corpus/labeling.md).

Codes are added only when verified against real DataJud records or CNJ SGT.
Unknown codes never guess: they fall through to organ rules or review.
"""

TPU_VERSION = "tpu-v1"
CODE_TO_AREA: dict[str, str] = {}


def map_code(code: str) -> str | None:
  """Area for one verified TPU code; None when unknown (never guessed)."""
  return CODE_TO_AREA.get(code)


def map_codes(codes: tuple[str, ...]) -> set[str]:
  """Distinct areas signaled by a record's TPU codes."""
  areas: set[str] = set()
  for code in codes:
    area = map_code(code)
    if area is not None:
      areas.add(area)
  return areas
