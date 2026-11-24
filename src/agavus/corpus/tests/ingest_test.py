"""Tests for ingestion orchestrators (tests folder, mock transport)."""

from datetime import date
from pathlib import Path

from agavus.config.settings import Settings
from agavus.corpus.httpfixtures import DATAJUD_HIT, json_handler, mock_client, text_handler
from agavus.corpus.ingest import import_statute_file, ingest_datajud, ingest_statutes
from agavus.corpus.storage import CorpusStorage


def _settings(tmp_path: Path) -> Settings:
  return Settings(
    name="world", data_dir=tmp_path, data_start=date(2020, 1, 1), datajud_api_key="test-key"
  )


def test_datajud_ingest_idempotent(tmp_path: Path) -> None:
  expected_runs = 2
  client = mock_client(json_handler({"hits": {"hits": [DATAJUD_HIT]}}))
  settings = _settings(tmp_path)
  assert ingest_datajud(settings, 10, client=client) == 1
  assert ingest_datajud(settings, 10, client=client) == 0
  storage = CorpusStorage(tmp_path)
  assert storage.watermark("datajud") == "2024-05-10"
  assert len(storage.load_manifest()["runs"]) == expected_runs


def test_datajud_empty_response_keeps_watermark(tmp_path: Path) -> None:
  client = mock_client(json_handler({"hits": {"hits": []}}))
  settings = _settings(tmp_path)
  assert ingest_datajud(settings, 10, client=client) == 0
  assert CorpusStorage(tmp_path).watermark("datajud") is None


def test_statutes_ingest_idempotent(tmp_path: Path) -> None:
  expected_new = 4  # two statutes x two articles
  html = "<p>Art. 1º Texto sintético da lei.</p><p>Art. 2º Outro texto.</p>"
  client = mock_client(text_handler(html))
  settings = _settings(tmp_path)
  assert ingest_statutes(settings, client=client) == expected_new
  assert ingest_statutes(settings, client=client) == 0


def test_offline_statute_import(tmp_path: Path) -> None:
  payload = tmp_path / "law.html"
  payload.write_text("<p>Art. 1º Texto offline.</p>", encoding="utf-8")
  settings = _settings(tmp_path)
  assert import_statute_file(settings, "Lei 12.318/2010", payload) == 1
  assert import_statute_file(settings, "Lei 12.318/2010", payload) == 0
  storage = CorpusStorage(tmp_path)
  assert storage.load_texts()[0].origin_ref == "Lei 12.318/2010, art. 1"
