"""Typed corpus records with mandatory provenance (docs/corpus/schema.md)."""

import hashlib
from dataclasses import dataclass
from datetime import date
from enum import Enum


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
