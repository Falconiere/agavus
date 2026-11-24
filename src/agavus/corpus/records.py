"""Typed corpus records with mandatory provenance (docs/corpus/schema.md)."""

import hashlib
from dataclasses import dataclass
from datetime import date
from enum import Enum

from agavus.corpus.jsonutil import as_dict, as_list, as_str, parse_date


class LabelSource(Enum):
  """Where a domain label came from; PENDING = not yet labeled."""

  TPU = "tpu"
  ORGAN = "organ"
  MANUAL = "manual"
  WEAK = "weak"
  PENDING = "pending"


class TextSource(Enum):
  """Origin system of a full-text document."""

  DJEN = "djen"
  PORTAL = "portal"
  STATUTE = "statute"


@dataclass(frozen=True)
class Movement:
  """One dated procedural event."""

  code: str
  name: str
  occurred_at: date


@dataclass(frozen=True)
class CaseRecord:
  """Case metadata from DataJud; id is the process number."""

  id: str
  tribunal: str
  orgao_julgador: str
  classe_tpu: str
  assuntos_tpu: tuple[str, ...]
  data_ajuizamento: date | None
  movimentos: tuple[Movement, ...]
  domain_label: str | None
  label_source: LabelSource
  raw_ref: str


@dataclass(frozen=True)
class TextDocument:
  """Full text (decision or statute) with a verbatim-citable reference."""

  id: str
  source: TextSource
  origin_ref: str
  published_at: date | None
  text: str
  citations: tuple[str, ...]
  domain_label: str | None
  label_source: LabelSource
  raw_ref: str


def text_document_id(source: TextSource, origin_ref: str) -> str:
  """Deterministic id: same source + reference always yields the same id."""
  digest = hashlib.sha256(f"{source.value}:{origin_ref}".encode()).hexdigest()
  return digest[:24]


def _common_fields(data: dict[str, object]) -> tuple[str | None, LabelSource, str]:
  """Shared provenance fields for record reconstruction."""
  return (
    as_str(data.get("domain_label")) or None,
    LabelSource(as_str(data.get("label_source")) or "pending"),
    as_str(data.get("raw_ref")),
  )


def case_from_dict(data: dict[str, object]) -> CaseRecord:
  """Reconstruct a CaseRecord from its stored JSON form."""
  movimentos: list[Movement] = []
  for item in as_list(data.get("movimentos")):
    entry = as_dict(item)
    occurred = parse_date(entry.get("occurred_at"))
    if occurred is not None:
      movimentos.append(
        Movement(
          code=as_str(entry.get("code")),
          name=as_str(entry.get("name")),
          occurred_at=occurred,
        )
      )
  domain_label, label_source, raw_ref = _common_fields(data)
  return CaseRecord(
    id=as_str(data.get("id")),
    tribunal=as_str(data.get("tribunal")),
    orgao_julgador=as_str(data.get("orgao_julgador")),
    classe_tpu=as_str(data.get("classe_tpu")),
    assuntos_tpu=tuple(as_str(item) for item in as_list(data.get("assuntos_tpu"))),
    data_ajuizamento=parse_date(data.get("data_ajuizamento")),
    movimentos=tuple(movimentos),
    domain_label=domain_label,
    label_source=label_source,
    raw_ref=raw_ref,
  )


def text_from_dict(data: dict[str, object]) -> TextDocument:
  """Reconstruct a TextDocument from its stored JSON form."""
  domain_label, label_source, raw_ref = _common_fields(data)
  return TextDocument(
    id=as_str(data.get("id")),
    source=TextSource(as_str(data.get("source")) or "statute"),
    origin_ref=as_str(data.get("origin_ref")),
    published_at=parse_date(data.get("published_at")),
    text=as_str(data.get("text")),
    citations=tuple(as_str(item) for item in as_list(data.get("citations"))),
    domain_label=domain_label,
    label_source=label_source,
    raw_ref=raw_ref,
  )
