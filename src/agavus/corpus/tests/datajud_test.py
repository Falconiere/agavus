"""Tests for DataJud query building and parsing (tests folder, synthetic)."""

from datetime import date

from agavus.corpus.datajud import (
  PAGE_MAX,
  auth_headers,
  build_query,
  hit_count,
  latest_filing,
  parse_case,
  parse_hits,
)
from agavus.corpus.httpfixtures import DATAJUD_HIT as HIT


def test_build_query_from_today_backward() -> None:
  page_size = 50
  query = build_query(date(2026, 11, 24), page_size)
  assert query["size"] == page_size
  assert query["sort"] == [{"dataAjuizamento": "desc"}, {"_id": "desc"}]
  boolean = query["query"]
  assert isinstance(boolean, dict)
  inner = boolean["bool"]
  assert isinstance(inner, dict)
  filters = inner["filter"]
  assert isinstance(filters, list)
  window = filters[0]
  assert isinstance(window, dict)
  bounds = window["range"]
  assert isinstance(bounds, dict)
  filed = bounds["dataAjuizamento"]
  assert isinstance(filed, dict)
  assert filed["lte"] == "20261124235959"
  assert "gte" not in filed
  resumed = build_query(date(2026, 11, 24), page_size, ("20261101010101", "abc"))
  assert resumed["search_after"] == ["20261101010101", "abc"]
  assert build_query(date(2026, 11, 24), PAGE_MAX + 1)["size"] == PAGE_MAX


def test_parse_case_fields() -> None:
  record = parse_case(HIT, "raw/datajud/x.json")
  assert record is not None
  assert record.id == "0701234-56.2024.8.07.0001"
  assert record.classe_tpu == "12345"
  assert record.assuntos_tpu == ("9999",)
  assert record.data_ajuizamento == date(2024, 5, 10)
  assert len(record.movimentos) == 1
  assert record.domain_label is None


def test_parse_case_missing_number_is_none() -> None:
  assert parse_case({"_source": {}}, "raw/x.json") is None


def test_parse_hits_skips_unusable() -> None:
  expected_hits = 2
  body: dict[str, object] = {"hits": {"hits": [HIT, {"_source": {}}]}}
  assert hit_count(body) == expected_hits
  records = parse_hits(body, "raw/x.json")
  assert len(records) == 1


def test_latest_filing() -> None:
  records = parse_hits({"hits": {"hits": [HIT]}}, "raw/x.json")
  assert latest_filing(records) == date(2024, 5, 10)
  assert latest_filing([]) is None


def test_auth_headers() -> None:
  headers = auth_headers("key-123")
  assert headers["Authorization"] == "APIKey key-123"
