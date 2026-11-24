"""DJEN (Comunica) ingestion: published TJDFT decisions on the F/M subject.

No start date. Page 1 of the public API is the newest communication
(documented by the CNJ portal; this host gets HTTP 403 from outside Brazil,
so the order is fixture-tested and assumed). Party fields are not stored.
"""

import json
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

from agavus.config.settings import Settings
from agavus.corpus.anonymize import anonymize
from agavus.corpus.batch import IngestBatch
from agavus.corpus.fetch import Fetcher, FetchPolicy, JsonRequest
from agavus.corpus.jsonutil import as_dict, as_str, parse_date
from agavus.corpus.records import LabelSource, TextDocument, TextSource, text_document_id
from agavus.corpus.span import span
from agavus.corpus.statutes import extract_citations
from agavus.corpus.storage import CorpusStorage
from agavus.corpus.subject import DJEN_TERMS

SOURCE = "djen"
BASE_URL = "https://comunicaapi.pje.jus.br/api/v1/comunicacao"
PAGE_SIZE = 50
TRIBUNAL = "TJDFT"
_DECISIONS = frozenset({"acordao", "decisao", "sentenca", "despacho"})
_FOLD = str.maketrans("áàãâéêíóôõúüç", "aaaaeeiooouuc")


@dataclass(frozen=True)
class TermState:
  """Next page to fetch, and whether the term has reached its oldest row."""

  page: int
  done: bool


def fold(text: str) -> str:
  """Accent-folded casefold, for the decision allow-list."""
  return text.casefold().translate(_FOLD)


def build_url(term: str, page: int) -> str:
  """One subject query. itensPorPagina above 50 comes back empty."""
  query = urlencode(
    {
      "siglaTribunal": TRIBUNAL,
      "texto": term,
      "pagina": str(page),
      "itensPorPagina": str(PAGE_SIZE),
    }
  )
  return f"{BASE_URL}?{query}"


def _items(body: dict[str, object]) -> list[dict[str, object]] | None:
  raw = body.get("items")
  if raw is None:
    raw = body.get("content")
  if not isinstance(raw, list):
    return None
  return [as_dict(item) for item in raw]


def _is_decision(item: dict[str, object]) -> bool:
  kind = as_str(item.get("tipoDocumento")) or as_str(item.get("tipoComunicacao"))
  return fold(kind) in _DECISIONS


def _origin(item: dict[str, object]) -> str:
  kind = as_str(item.get("tipoDocumento")) or as_str(item.get("tipoComunicacao")) or "Decisão"
  organ = as_str(item.get("nomeOrgao")) or "órgão não informado"
  number = (
    as_str(item.get("numeroprocessocommascara"))
    or as_str(item.get("numero_processo"))
    or as_str(item.get("numeroProcesso"))
    or "s/n"
  )
  published = as_str(item.get("data_disponibilizacao")) or as_str(item.get("dataDisponibilizacao"))
  ident = as_str(item.get("id")) or as_str(item.get("hash"))
  return f"{kind}, {organ}, processo {number}, DJEN {published}, id {ident}"


def _document(item: dict[str, object], raw_ref: str) -> TextDocument | None:
  if not _is_decision(item):
    return None
  text = anonymize(as_str(item.get("texto"))).strip()
  if not text:
    return None
  origin = _origin(item)
  published = parse_date(item.get("data_disponibilizacao") or item.get("dataDisponibilizacao"))
  return TextDocument(
    id=text_document_id(TextSource.DJEN, origin),
    source=TextSource.DJEN,
    origin_ref=origin,
    published_at=published,
    text=text,
    citations=extract_citations(text),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref=raw_ref,
  )


def parse_documents(body: dict[str, object], raw_ref: str) -> list[TextDocument] | None:
  """Decision documents in response order. None when the body has no item list."""
  items = _items(body)
  if items is None:
    return None
  return [doc for item in items if (doc := _document(item, raw_ref)) is not None]


def encode_terms(states: dict[str, TermState]) -> str:
  """Per-term watermark. Page 1 is the newest page."""
  payload = {
    term: {"done": state.done, "page": state.page} for term, state in sorted(states.items())
  }
  return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def decode_terms(raw: str | None) -> dict[str, TermState]:
  """Load term cursors. Unknown terms start at page 1."""
  stored: dict[str, object] = {}
  if raw and raw.startswith("{"):
    try:
      loaded = json.loads(raw)
    except json.JSONDecodeError:
      loaded = {}
    if isinstance(loaded, dict):
      stored = loaded
  states: dict[str, TermState] = {}
  for term in DJEN_TERMS:
    entry = as_dict(stored.get(term))
    page = entry.get("page")
    number = page if isinstance(page, int) and page > 0 else 1
    states[term] = TermState(number, entry.get("done") is True)
  return states


def _fetch(
  fetcher: Fetcher, term: str, page: int
) -> tuple[dict[str, object], list[dict[str, object]] | None]:
  body = fetcher.fetch_json(JsonRequest("GET", build_url(term, page), name="comunicacao"))
  return body, _items(body)


def _save_new(storage: CorpusStorage, documents: list[TextDocument]) -> int:
  fresh: list[TextDocument] = []
  for document in documents:
    if storage.load_text_by_origin_ref(document.origin_ref) is None:
      fresh.append(document)
  return storage.save_texts(fresh)


def _commit(storage: CorpusStorage, states: dict[str, TermState]) -> None:
  storage.set_watermark(SOURCE, encode_terms(states))


def _walk_term(
  storage: CorpusStorage,
  fetcher: Fetcher,
  states: dict[str, TermState],
  term: str,
  pages_left: int | None,
) -> tuple[int, int]:
  """Continue one term toward older pages. Returns new rows and pages used."""
  new = 0
  used = 0
  state = states[term]
  while not state.done and (pages_left is None or used < pages_left):
    _body, items = _fetch(fetcher, term, state.page)
    if items is None:
      break
    documents = parse_documents({"items": items}, fetcher.last_raw_ref) or []
    added = storage.save_texts(documents)
    new += added
    short = len(items) < PAGE_SIZE
    state = TermState(state.page, True) if short else TermState(state.page + 1, False)
    states[term] = state
    _record(storage, term, added, documents)
    _commit(storage, states)
    used += 1
  return new, used


def _refresh_term(
  storage: CorpusStorage,
  fetcher: Fetcher,
  states: dict[str, TermState],
  term: str,
) -> int:
  """Re-read the newest page. A full page of unseen decisions resumes the walk."""
  _body, items = _fetch(fetcher, term, 1)
  if items is None:
    return 0
  documents = parse_documents({"items": items}, fetcher.last_raw_ref) or []
  added = _save_new(storage, documents)
  resume = bool(documents) and added == len(documents) and len(items) >= PAGE_SIZE
  states[term] = TermState(2, False) if resume else TermState(states[term].page, True)
  _record(storage, term, added, documents)
  _commit(storage, states)
  return added


def _record(
  storage: CorpusStorage,
  term: str,
  new: int,
  documents: list[TextDocument],
) -> None:
  published = [document.published_at for document in documents if document.published_at]
  start, end = span(published)
  storage.record_run(f"{SOURCE}:{term}", len(documents), new, start, end)


def ingest_djen(
  settings: Settings,
  *,
  client: httpx.Client | None = None,
  policy: FetchPolicy | None = None,
  pages: int | None = None,
) -> IngestBatch:
  """Walk each subject term from the newest publication until it is exhausted."""
  storage = CorpusStorage(settings.data_dir)
  fetcher = Fetcher(storage, SOURCE, client=client, policy=policy)
  states = decode_terms(storage.watermark(SOURCE))
  new = 0
  left = pages
  for term in DJEN_TERMS:
    if left is not None and left <= 0:
      break
    if states[term].done:
      added = _refresh_term(storage, fetcher, states, term)
      used = 1
      if not states[term].done and (left is None or left > used):
        budget = None if left is None else left - used
        more, walked = _walk_term(storage, fetcher, states, term, budget)
        added += more
        used += walked
    else:
      added, used = _walk_term(storage, fetcher, states, term, left)
    new += added
    if left is not None:
      left -= used
  done = all(item.done for item in states.values())
  finished = sum(item.done for item in states.values())
  return IngestBatch(new, done, f"{finished}/{len(DJEN_TERMS)} terms done")
