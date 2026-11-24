"""Ingestion orchestrators: fetch → parse → store → manifest → watermark."""

from datetime import date
from pathlib import Path

import httpx

from agavus.config.settings import Settings
from agavus.corpus import datajud, statutes
from agavus.corpus.fetch import Fetcher, JsonRequest
from agavus.corpus.storage import CorpusStorage


def ingest_datajud(settings: Settings, limit: int, *, client: httpx.Client | None = None) -> int:
  """Fetch one window page of TJDFT metadata; return count of new records.

  Incremental: starts at the stored watermark when present. The watermark
  advances only when records were actually parsed (empty/overloaded responses
  never move it). Re-fetch overlap is absorbed by idempotent storage.
  """
  storage = CorpusStorage(settings.data_dir)
  fetcher = Fetcher(storage, datajud.SOURCE, client=client)
  end = date.today()
  start = settings.data_start
  watermark = storage.watermark(datajud.SOURCE)
  if watermark:
    start = max(start, date.fromisoformat(watermark))
  body = fetcher.fetch_json(
    JsonRequest(
      "POST",
      datajud.BASE_URL,
      headers=datajud.auth_headers(settings.datajud_api_key),
      payload=datajud.build_query(start, end, limit),
      name="search",
    )
  )
  records = datajud.parse_hits(body, fetcher.last_raw_ref)
  new = storage.save_cases(records)
  storage.record_run(datajud.SOURCE, datajud.hit_count(body), new, start, end)
  latest = datajud.latest_filing(records)
  if latest is not None:
    storage.set_watermark(datajud.SOURCE, latest.isoformat())
  return new


def ingest_statutes(settings: Settings, *, client: httpx.Client | None = None) -> int:
  """Fetch current consolidated statutes; one record per article."""
  storage = CorpusStorage(settings.data_dir)
  fetcher = Fetcher(storage, statutes.SOURCE, client=client)
  today = date.today()
  fetched = 0
  new = 0
  for statute in statutes.STATUTES:
    payload = fetcher.fetch(JsonRequest("GET", statute.url, name="statute"))
    html = statutes.decode_html(payload)
    documents = statutes.parse_statute(
      statute.name, html, fetcher.last_raw_ref, statute.article_range
    )
    fetched += len(documents)
    new += storage.save_texts(documents)
  storage.record_run(statutes.SOURCE, fetched, new, today, today)
  return new


def import_statute_file(settings: Settings, name: str, path: Path) -> int:
  """Offline import of an already-fetched statute page (raw preserved).

  For networks where direct egress is blocked: fetch via any authorized
  tool, then import the payload. Same parsing, provenance, and idempotency
  as the online path.
  """
  storage = CorpusStorage(settings.data_dir)
  payload = path.read_bytes()
  today = date.today()
  raw_ref = storage.save_raw(
    statutes.SOURCE,
    today,
    f"{name.replace('/', '-')}.html",
    payload,
    {"source": "offline-file", "path": str(path), "name": name},
  )
  article_range = next(
    (statute.article_range for statute in statutes.STATUTES if statute.name == name), None
  )
  documents = statutes.parse_statute(name, statutes.decode_html(payload), raw_ref, article_range)
  new = storage.save_texts(documents)
  storage.record_run(statutes.SOURCE, len(documents), new, today, today)
  return new
