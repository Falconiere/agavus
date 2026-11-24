"""Tests for ingestion orchestrators (tests folder, mock transport)."""

import json
from pathlib import Path

import httpx

from agavus.config.settings import Settings
from agavus.corpus.cursor import decode
from agavus.corpus.fetch import FetchPolicy
from agavus.corpus.httpfixtures import (
  DATAJUD_HIT,
  json_handler,
  mock_client,
  sample_settings,
  text_handler,
)
from agavus.corpus.ingest import import_statute_file, ingest_datajud, ingest_statutes
from agavus.corpus.storage import CorpusStorage

FAST = FetchPolicy(min_interval=0, max_retries=0)


def _settings(tmp_path: Path) -> Settings:
  return sample_settings(tmp_path)


def _hit(number: str, filed: str, sort_id: str, *, sigilo: int = 0) -> dict[str, object]:
  source = dict(DATAJUD_HIT["_source"]) if isinstance(DATAJUD_HIT["_source"], dict) else {}
  source = dict(source)
  source["numeroProcesso"] = number
  source["dataAjuizamento"] = filed
  source["nivelSigilo"] = sigilo
  return {"_source": source, "sort": [filed, sort_id], "_id": sort_id}


def test_datajud_ingest_idempotent(tmp_path: Path) -> None:
  hit = _hit("0701234-56.2024.8.07.0001", "20240510120000", "a")
  client = mock_client(json_handler({"hits": {"hits": [hit]}}))
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


def test_datajud_walks_newer_then_older(tmp_path: Path) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content.decode())
    filed = payload["query"]["bool"]["filter"][0]["range"]["dataAjuizamento"]
    assert str(filed["lte"]).endswith("235959")
    assert "gte" not in filed
    assert payload["sort"][0] == {"dataAjuizamento": "desc"}
    after = payload.get("search_after")
    if after == ["20200102000000", "old"]:
      return httpx.Response(
        200, json={"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}}
      )
    if after == ["20240510120000", "new"]:
      hits = [_hit("0700001-56.2020.8.07.0001", "20200102000000", "old")]
    else:
      assert after is None
      hits = [_hit("0701234-56.2024.8.07.0001", "20240510120000", "new")]
    return httpx.Response(200, json={"hits": {"hits": hits}})

  settings = _settings(tmp_path)
  batch = ingest_datajud(settings, 1, client=mock_client(handler), policy=FAST)
  storage = CorpusStorage(tmp_path)
  assert batch.done
  stored = {case.id for case in storage.load_cases()}
  assert stored == {
    "0701234-56.2024.8.07.0001",
    "0700001-56.2020.8.07.0001",
  }


def test_datajud_empty_response_keeps_watermark(tmp_path: Path) -> None:
  client = mock_client(json_handler({"hits": {"hits": []}}))
  settings = _settings(tmp_path)
  batch = ingest_datajud(settings, 10, client=client, policy=FAST)
  assert batch.new == 0
  assert not batch.done
  assert CorpusStorage(tmp_path).watermark("datajud") is None


def test_datajud_skips_sealed_metadata(tmp_path: Path) -> None:
  sealed = _hit("0701234-56.2024.8.07.0001", "20240510120000", "a", sigilo=1)
  body: dict[str, object] = {"hits": {"hits": [sealed]}}
  client = mock_client(json_handler(body))
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
