"""Tests for DJEN ingestion (tests folder, mock transport, no network)."""

from pathlib import Path

import httpx

from agavus.corpus.djen import decode_terms, ingest_djen
from agavus.corpus.fetch import FetchPolicy
from agavus.corpus.httpfixtures import mock_client, sample_settings
from agavus.corpus.records import TextSource
from agavus.corpus.storage import CorpusStorage

FAST = FetchPolicy(min_interval=0, max_retries=0)
_ITEM = {
  "id": "9",
  "tipoDocumento": "Decisão",
  "nomeOrgao": "1ª Vara de Família",
  "numeroprocessocommascara": "0701234-56.2024.8.07.0001",
  "data_disponibilizacao": "2024-05-10",
  "texto": "A autora Maria Silva pede guarda. Art. 2º, Lei 12.318/2010.",
}


def test_djen_stores_decision_and_skips_notice(tmp_path: Path) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    page = request.url.params["pagina"]
    term = request.url.params["texto"]
    assert request.url.params["siglaTribunal"] == "TJDFT"
    if term != "violência doméstica":
      return httpx.Response(200, json={"items": []})
    if page == "1":
      notice = {"id": "1", "tipoDocumento": "Intimação", "nomeOrgao": "Vara", "texto": "cite-se"}
      return httpx.Response(200, json={"items": [_ITEM, notice]})
    return httpx.Response(200, json={"items": []})

  settings = sample_settings(tmp_path)
  first = ingest_djen(settings, client=mock_client(handler), policy=FAST)
  second = ingest_djen(settings, client=mock_client(handler), policy=FAST)
  storage = CorpusStorage(tmp_path)
  texts = storage.load_texts()
  assert first.new == 1
  assert first.done
  assert second.new == 0
  assert len(texts) == 1
  assert texts[0].source is TextSource.DJEN
  assert "Maria Silva" not in texts[0].text
  assert "[nome removido]" in texts[0].text
  assert "Vara de Família" in texts[0].origin_ref
  assert all(state.done for state in decode_terms(storage.watermark("djen")).values())
