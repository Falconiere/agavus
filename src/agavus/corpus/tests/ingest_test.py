"""Tests for ingestion orchestrators (tests folder, mock transport)."""

import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from agavus.config.settings import Settings
from agavus.corpus import datajud
from agavus.corpus.cursor import decode
from agavus.corpus.fetch import FetchPolicy
from agavus.corpus.httpfixtures import (
  DATAJUD_HIT,
  Handler,
  json_handler,
  mock_client,
  sample_settings,
  text_handler,
)
from agavus.corpus.ingest import import_statute_file, ingest_datajud, ingest_statutes
from agavus.corpus.storage import CorpusStorage

FAST = FetchPolicy(min_interval=0, max_retries=0)
_WHEN = date(2024, 5, 20)


def _pin_clock(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(datajud, "clock", lambda: _WHEN)
  monkeypatch.setattr(datajud, "EMPTY_MONTHS", 2)


def _settings(tmp_path: Path) -> Settings:
  return sample_settings(tmp_path)


def _hit(number: str, filed: str, sort_id: str, *, sigilo: int = 0) -> dict[str, object]:
  source = dict(DATAJUD_HIT["_source"]) if isinstance(DATAJUD_HIT["_source"], dict) else {}
  source = dict(source)
  source["numeroProcesso"] = number
  source["dataAjuizamento"] = filed
  source["nivelSigilo"] = sigilo
  return {"_source": source, "sort": [filed, sort_id], "_id": sort_id}


def _empty_page() -> httpx.Response:
  body: dict[str, object] = {"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}}
  return httpx.Response(200, json=body)


def _decoded(request: httpx.Request) -> tuple[dict[str, object], dict[str, object]]:
  parsed = json.loads(request.content.decode())
  payload: dict[str, object] = parsed if isinstance(parsed, dict) else {}
  query = payload.get("query")
  boolean = query.get("bool") if isinstance(query, dict) else None
  filters = boolean.get("filter") if isinstance(boolean, dict) else None
  first = filters[0] if isinstance(filters, list) and filters else None
  window = first.get("range") if isinstance(first, dict) else None
  filed = window.get("dataAjuizamento") if isinstance(window, dict) else None
  return payload, filed if isinstance(filed, dict) else {}


def _in_may(request: httpx.Request) -> bool:
  _payload, filed = _decoded(request)
  return str(filed.get("gte", "")).startswith("202405")


def _may_only(hit: dict[str, object]) -> Handler:
  def handler(request: httpx.Request) -> httpx.Response:
    if not _in_may(request):
      return _empty_page()
    return httpx.Response(200, json={"hits": {"hits": [hit]}})

  return handler


def _month_handler(request: httpx.Request) -> httpx.Response:
  payload, filed = _decoded(request)
  assert str(filed.get("lte", "")).endswith("235959")
  assert "gte" in filed
  assert payload.get("sort") == [{"dataAjuizamento": "desc"}, {"_id": "desc"}]
  after = payload.get("search_after")
  if not _in_may(request) or after == ["20240502120000", "old"]:
    return _empty_page()
  if after == ["20240510120000", "new"]:
    hits = [_hit("0700001-56.2024.8.07.0001", "20240502120000", "old")]
  else:
    hits = [_hit("0701234-56.2024.8.07.0001", "20240510120000", "new")]
  return httpx.Response(200, json={"hits": {"hits": hits}})


def test_datajud_ingest_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  _pin_clock(monkeypatch)
  hit = _hit("0701234-56.2024.8.07.0001", "20240510120000", "a")
  client = mock_client(_may_only(hit))
  settings = _settings(tmp_path)
  first = ingest_datajud(settings, 10, client=client, policy=FAST)
  second = ingest_datajud(settings, 10, client=client, policy=FAST)
  assert first.new == 1
  assert first.done
  assert second.new == 0
  assert second.done
  storage = CorpusStorage(tmp_path)
  state = decode(storage.watermark("datajud"))
  assert state.done
  assert state.newest == ("20240510120000", "a")
  assert len(storage.load_manifest()["runs"]) == 1


def test_datajud_walks_newer_then_older(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  _pin_clock(monkeypatch)
  settings = _settings(tmp_path)
  batch = ingest_datajud(settings, 1, client=mock_client(_month_handler), policy=FAST)
  storage = CorpusStorage(tmp_path)
  assert batch.done
  stored = {case.id for case in storage.load_cases()}
  assert stored == {
    "0701234-56.2024.8.07.0001",
    "0700001-56.2024.8.07.0001",
  }


def test_datajud_empty_response_keeps_watermark(tmp_path: Path) -> None:
  client = mock_client(json_handler({"hits": {"hits": []}}))
  settings = _settings(tmp_path)
  batch = ingest_datajud(settings, 10, client=client, policy=FAST)
  assert batch.new == 0
  assert not batch.done
  assert CorpusStorage(tmp_path).watermark("datajud") is None


def test_datajud_skips_sealed_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  _pin_clock(monkeypatch)
  sealed = _hit("0701234-56.2024.8.07.0001", "20240510120000", "a", sigilo=1)
  client = mock_client(_may_only(sealed))
  batch = ingest_datajud(_settings(tmp_path), 10, client=client, policy=FAST)
  assert batch.new == 0
  assert batch.done
  assert CorpusStorage(tmp_path).load_cases() == []


def test_statutes_ingest_idempotent(tmp_path: Path) -> None:
  expected_new = 8  # four statutes x two articles (CC range excludes arts. 1-2)
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
