"""Inclusive date span of a page, for the ingestion manifest."""

from datetime import date


def span(dates: list[date]) -> tuple[date, date]:
  """Earliest and latest date, or today when the page has none."""
  if not dates:
    today = date.today()
    return today, today
  return min(dates), max(dates)
