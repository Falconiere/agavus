"""DataJud ingestion: TJDFT case metadata (docs/corpus/sources.md).

The public API is frequently overloaded; the Fetcher handles patience.
TPU codes are stored as-is — labeling happens in the taxonomy phase.
"""

from datetime import date

from agavus.corpus.jsonutil import as_dict, as_list, as_str, parse_date
from agavus.corpus.records import CaseRecord, LabelSource, Movement

SOURCE = "datajud"
BASE_URL = "https://api-publica.datajud.cnj.jus.br/api_publica_tjdft/_search"


def build_query(start: date, end: date, size: int) -> dict[str, object]:
  """Window query: TJDFT cases filed between start and end (inclusive)."""
  return {
    "size": size,
    "query": {"range": {"dataAjuizamento": {"gte": start.isoformat(), "lte": end.isoformat()}}},
    "sort": [{"dataAjuizamento": "asc"}],
  }


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
  "SOURCE",
  "auth_headers",
  "build_query",
  "hit_count",
  "latest_filing",
  "parse_case",
  "parse_hits",
]
