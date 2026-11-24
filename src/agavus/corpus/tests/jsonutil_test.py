"""Tests for JSON navigation helpers (tests folder)."""

from datetime import date

from agavus.corpus.jsonutil import as_dict, as_int, as_list, as_str, parse_date


def test_as_dict() -> None:
  assert as_dict({"a": 1}) == {"a": 1}
  assert as_dict(None) == {}
  assert as_dict([1]) == {}


def test_as_list() -> None:
  assert as_list([1, 2]) == [1, 2]
  assert as_list("x") == []


def test_as_str_and_int() -> None:
  sample_int = 3
  assert as_str("ok") == "ok"
  assert as_str(sample_int) == ""
  assert as_int(sample_int) == sample_int
  assert as_int("3") == 0


def test_parse_date_variants() -> None:
  assert parse_date("2024-05-10") == date(2024, 5, 10)
  assert parse_date("2024-05-10T14:30:00.000Z") == date(2024, 5, 10)
  assert parse_date("20240510143000") == date(2024, 5, 10)
  assert parse_date("20241301000000") is None
  assert parse_date("") is None
  assert parse_date("not-a-date") is None
  assert parse_date(None) is None
