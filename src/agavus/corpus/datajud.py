"""DataJud ingestion: TJDFT case metadata for the F/M subject (docs/corpus/sources.md).

The public API is frequently overloaded; the Fetcher handles patience.
There is no start date: queries are bounded only by the end of today and
walk newest-first. TPU codes are stored as-is — labeling is a later pass.
`numeroProcesso` is a text field and cannot be a sort key (the API returns
400). The tie-break is `_id`.
"""

from dataclasses import dataclass
from datetime import date

from agavus.corpus.cursor import WalkState, comes_before
from agavus.corpus.jsonutil import as_dict, as_int, as_list, as_str, parse_date
from agavus.corpus.records import CaseRecord, LabelSource, Movement
from agavus.corpus.subject import DATAJUD_FIELDS, SUBJECT_PHRASES
from agavus.taxonomy.tpu import CODE_TO_AREA

SOURCE = "datajud"
BASE_URL = "https://api-publica.datajud.cnj.jus.br/api_publica_tjdft/_search"
PAGE_MAX = 10_000
UNKNOWN_LIMIT = 20


def clamp_size(size: int) -> int:
  """Elasticsearch page size accepted by the public API (1..10000)."""
  return min(PAGE_MAX, max(1, size))


def upper_bound(end: date) -> str:
  """Compact end-of-day. An ISO upper bound matches nothing on this index."""
  return end.strftime("%Y%m%d") + "235959"


def build_query(
  end: date, size: int, search_after: tuple[str, ...] | None = None
) -> dict[str, object]:
  """Subject query from today backward. No lower bound."""
  should: list[object] = [
    {"match_phrase": {field: phrase}} for phrase in SUBJECT_PHRASES for field in DATAJUD_FIELDS
  ]
  body: dict[str, object] = {
    "size": clamp_size(size),
    "query": {
      "bool": {
        "filter": [{"range": {"dataAjuizamento": {"lte": upper_bound(end)}}}],
        "should": should,
        "minimum_should_match": 1,
      }
    },
    "sort": [{"dataAjuizamento": "desc"}, {"_id": "desc"}],
  }
  if search_after:
    body["search_after"] = list(search_after)
  return body


def _movements(raw: object) -> tuple[Movement, ...]:
  movements: list[Movement] = []
  for item in as_list(raw):
    entry = as_dict(item)
    occurred = parse_date(entry.get("dataHora"))
    if occurred is not None:
      movements.append(
        Movement(
          code=as_str(entry.get("codigo")), name=as_str(entry.get("nome")), occurred_at=occurred
        )
      )
  return tuple(movements)


def parse_case(hit: dict[str, object], raw_ref: str) -> CaseRecord | None:
  """Parse one Elasticsearch hit into a CaseRecord; None when unusable."""
  source = as_dict(hit.get("_source"))
  number = as_str(source.get("numeroProcesso"))
  if not number:
    return None
  classe = as_dict(source.get("classe"))
  assuntos = tuple(
    as_str(as_dict(item).get("codigo")) or as_str(as_dict(item).get("nome"))
    for item in as_list(source.get("assuntos"))
  )
  return CaseRecord(
    id=number,
    tribunal=as_str(source.get("tribunal")) or "TJDFT",
    orgao_julgador=as_str(as_dict(source.get("orgaoJulgador")).get("nome")),
    classe_tpu=as_str(classe.get("codigo")) or as_str(classe.get("nome")),
    assuntos_tpu=assuntos,
    data_ajuizamento=parse_date(source.get("dataAjuizamento")),
    movimentos=_movements(source.get("movimentos")),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref=raw_ref,
  )


def latest_filing(records: list[CaseRecord]) -> date | None:
  """Newest filing date among records — the next incremental watermark."""
  dates = [r.data_ajuizamento for r in records if r.data_ajuizamento is not None]
  return max(dates) if dates else None


def auth_headers(api_key: str) -> dict[str, str]:
  """DataJud public-API authorization header."""
  return {"Authorization": f"APIKey {api_key}", "Content-Type": "application/json"}


def hit_count(body: dict[str, object]) -> int:
  """Number of hits in a search response (tolerates partial/empty results)."""
  return len(as_list(as_dict(body.get("hits")).get("hits")))


def _sort_token(value: object) -> str:
  if isinstance(value, str):
    return value
  if isinstance(value, int):
    return str(value)
  return ""


def hit_sort_key(hit: dict[str, object]) -> tuple[str, ...]:
  """Sort tuple to echo as search_after. Falls back to source fields."""
  raw = hit.get("sort")
  if isinstance(raw, list) and raw:
    tokens = tuple(_sort_token(item) for item in raw)
    if all(tokens):
      return tokens
  source = as_dict(hit.get("_source"))
  return (as_str(source.get("dataAjuizamento")), as_str(hit.get("_id")))


def _sigilo(source: dict[str, object]) -> int:
  level = source.get("nivelSigilo")
  if isinstance(level, int):
    return level
  text = as_str(level)
  return int(text) if text.isdigit() else 0


def _subjects(source: dict[str, object]) -> list[tuple[str, str]]:
  pairs: list[tuple[str, str]] = []
  classe = as_dict(source.get("classe"))
  code = as_str(classe.get("codigo"))
  if code:
    pairs.append((code, as_str(classe.get("nome"))))
  for item in as_list(source.get("assuntos")):
    entry = as_dict(item)
    assunto = as_str(entry.get("codigo"))
    if assunto:
      pairs.append((assunto, as_str(entry.get("nome"))))
  return pairs


def exhausted(body: dict[str, object]) -> bool:
  """True only for a genuine empty page (total 0), not a timed-out or partial body."""
  if body.get("timed_out") is True or hit_count(body) > 0:
    return False
  shards = as_dict(body.get("_shards"))
  if as_int(shards.get("failed")) > 0:
    return False
  total = as_dict(as_dict(body.get("hits")).get("total"))
  return as_int(total.get("value")) == 0 and as_str(total.get("relation")) == "eq"


@dataclass(frozen=True)
class Taken:
  """One response, trimmed to the walk direction."""

  records: tuple[CaseRecord, ...]
  subjects: tuple[tuple[str, str], ...]
  newest_key: tuple[str, ...] | None
  last_key: tuple[str, ...] | None
  full: bool
  exhausted: bool
  reached_frontier: bool


def take(
  body: dict[str, object],
  raw_ref: str,
  size: int,
  frontier: tuple[str, ...] | None,
) -> Taken:
  """Parse a page. With a frontier, stop at the first hit that is not newer."""
  hits = [as_dict(hit) for hit in as_list(as_dict(body.get("hits")).get("hits"))]
  if exhausted(body):
    return Taken((), (), None, None, False, True, False)
  records: list[CaseRecord] = []
  subjects: list[tuple[str, str]] = []
  newest: tuple[str, ...] | None = None
  last: tuple[str, ...] | None = None
  reached = False
  for hit in hits:
    key = hit_sort_key(hit)
    if frontier is not None and not comes_before(key, frontier):
      reached = True
      break
    newest = key if newest is None else newest
    last = key
    source = as_dict(hit.get("_source"))
    subjects.extend(_subjects(source))
    if _sigilo(source) > 0:
      continue
    record = parse_case(hit, raw_ref)
    if record is not None:
      records.append(record)
  full = len(hits) >= size and not reached
  return Taken(tuple(records), tuple(subjects), newest, last, full, False, reached)


def advance_open(state: WalkState, taken: Taken) -> WalkState:
  """Move the historical cursor. A short page finishes the walk."""
  newest = state.newest if state.newest is not None else taken.newest_key
  if taken.full and taken.last_key is not None:
    return WalkState(taken.last_key, newest, False)
  return WalkState(None, newest, True)


def format_unknown(subjects: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
  """Top TPU codes on this batch that are not in the curated table."""
  counts: dict[tuple[str, str], int] = {}
  for code, name in subjects:
    if code in CODE_TO_AREA:
      continue
    counts[(code, name)] = counts.get((code, name), 0) + 1
  ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
  return tuple(f"{code} {name} {count}".strip() for (code, name), count in ranked[:UNKNOWN_LIMIT])


def parse_hits(body: dict[str, object], raw_ref: str) -> list[CaseRecord]:
  """Parse all usable hits from a search response."""
  records: list[CaseRecord] = []
  for hit in as_list(as_dict(body.get("hits")).get("hits")):
    record = parse_case(as_dict(hit), raw_ref)
    if record is not None:
      records.append(record)
  return records


__all__ = [
  "BASE_URL",
  "PAGE_MAX",
  "SOURCE",
  "Taken",
  "advance_open",
  "auth_headers",
  "build_query",
  "clamp_size",
  "exhausted",
  "format_unknown",
  "hit_count",
  "hit_sort_key",
  "latest_filing",
  "parse_case",
  "parse_hits",
  "take",
  "upper_bound",
]
