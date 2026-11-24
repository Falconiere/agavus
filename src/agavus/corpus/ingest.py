"""Ingestion orchestrators: fetch → parse → store → manifest → watermark."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

from agavus.config.settings import Settings
from agavus.corpus import datajud, statutes
from agavus.corpus.batch import IngestBatch
from agavus.corpus.cursor import WalkState, decode, encode
from agavus.corpus.fetch import Fetcher, FetchPolicy, JsonRequest
from agavus.corpus.jsonutil import parse_date
from agavus.corpus.span import span
from agavus.corpus.storage import CorpusStorage


@dataclass(frozen=True)
class _Pull:
  """One DataJud walk: shared clients plus the page budget."""

  storage: CorpusStorage
  fetcher: Fetcher
  settings: Settings
  limit: int
  pages: int | None


def _fetch_page(
  pull: _Pull, after: tuple[str, ...] | None, start: date, end: date
) -> dict[str, object]:
  return pull.fetcher.fetch_json(
    JsonRequest(
      "POST",
      datajud.BASE_URL,
      headers=datajud.auth_headers(pull.settings.datajud_api_key),
      payload=datajud.build_query(end, pull.limit, after, start),
      name="search",
    )
  )


def _note(storage: CorpusStorage, state: WalkState, taken: datajud.Taken, new: int) -> None:
  filed = [record.data_ajuizamento for record in taken.records if record.data_ajuizamento]
  start, end = span(filed)
  storage.set_watermark(datajud.SOURCE, encode(state))
  storage.record_run(datajud.SOURCE, len(taken.records), new, start, end)


def _caught_up(taken: datajud.Taken) -> bool:
  return taken.reached_frontier or taken.exhausted or not taken.full or taken.last_key is None


def _begin(state: WalkState) -> WalkState:
  if state.month is not None:
    return state
  return WalkState(state.after, state.newest, state.done, datajud.month_of(datajud.clock()), 0)


def _blank_month(pull: _Pull, state: WalkState) -> WalkState:
  state = datajud.finish_month(state, had_hits=False)
  pull.storage.set_watermark(datajud.SOURCE, encode(state))
  return state


def _walk_history(
  pull: _Pull, state: WalkState
) -> tuple[WalkState, int, tuple[tuple[str, str], ...]]:
  """One month at a time, newest first. An empty month is a cheap request."""
  state = _begin(state)
  new = 0
  seen = 0
  blanks = 0
  subjects: list[tuple[str, str]] = []
  size = datajud.clamp_size(pull.limit)
  while not state.done and (pull.pages is None or seen < pull.pages):
    today = datajud.clock()
    month = state.month or datajud.month_of(today)
    start, end = datajud.month_window(month, today)
    body = _fetch_page(pull, state.after, start, end)
    taken = datajud.take(body, pull.fetcher.last_raw_ref, size, None)
    seen += 1
    if taken.last_key is None and not taken.exhausted:
      break
    if taken.last_key is None and state.after is None:
      state = _blank_month(pull, state)
      blanks += 1
      if state.done or blanks >= datajud.EMPTY_MONTHS:
        break
      continue
    if taken.last_key is None:
      state = datajud.finish_month(state, had_hits=True)
      _note(pull.storage, state, taken, 0)
      continue
    blanks = 0
    added = pull.storage.save_cases(taken.records)
    state = datajud.advance_open(state, taken)
    if not taken.full:
      state = datajud.finish_month(state, had_hits=True)
    _note(pull.storage, state, taken, added)
    new += added
    subjects.extend(taken.subjects)
  return state, new, tuple(subjects)


def _refresh_newer(
  pull: _Pull, state: WalkState
) -> tuple[WalkState, int, tuple[tuple[str, str], ...]]:
  """A finished walk only pulls hits newer than the stored frontier."""
  if state.newest is None:
    return state, 0, ()
  new = 0
  seen = 0
  after: tuple[str, ...] | None = None
  head: tuple[str, ...] | None = None
  subjects: list[tuple[str, str]] = []
  size = datajud.clamp_size(pull.limit)
  origin = parse_date(state.newest[0])
  start = origin if origin is not None else datajud.clock()
  while pull.pages is None or seen < pull.pages:
    body = _fetch_page(pull, after, start, datajud.clock())
    taken = datajud.take(body, pull.fetcher.last_raw_ref, size, state.newest)
    added = pull.storage.save_cases(taken.records)
    new += added
    subjects.extend(taken.subjects)
    seen += 1
    if head is None and taken.newest_key is not None:
      head = taken.newest_key
    if _caught_up(taken):
      if head is not None:
        state = WalkState(None, head, True, state.month, 0)
        _note(pull.storage, state, taken, added)
      break
    after = taken.last_key
    _note(pull.storage, state, taken, added)
  return state, new, tuple(subjects)


def _cursor_label(state: WalkState) -> str:
  if state.done and state.newest is not None:
    return f"done newest={state.newest[0]}"
  if state.after is not None:
    return f"after={state.after[0]}"
  return "start"


def ingest_datajud(
  settings: Settings,
  limit: int,
  *,
  pages: int | None = None,
  client: httpx.Client | None = None,
  policy: FetchPolicy | None = None,
) -> IngestBatch:
  """Walk TJDFT F/M metadata from today backward until the source is exhausted.

  `pages` caps the run. The cursor is committed after every saved page.
  `settings.data_start` is ignored. A finished cursor only refreshes the head.
  """
  storage = CorpusStorage(settings.data_dir)
  fetcher = Fetcher(storage, datajud.SOURCE, client=client, policy=policy)
  pull = _Pull(storage, fetcher, settings, limit, pages)
  state = decode(storage.watermark(datajud.SOURCE))
  if state.done:
    state, new, subjects = _refresh_newer(pull, state)
  else:
    state, new, subjects = _walk_history(pull, state)
  return IngestBatch(new, state.done, _cursor_label(state), datajud.format_unknown(subjects))


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
