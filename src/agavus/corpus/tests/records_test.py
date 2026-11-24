"""Tests for corpus records (tests folder, synthetic data only)."""

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from agavus.corpus.records import (
  CaseRecord,
  LabelSource,
  Movement,
  TextDocument,
  TextSource,
  text_document_id,
)


def _case() -> CaseRecord:
  return CaseRecord(
    id="0701234-56.2024.8.07.0001",
    tribunal="TJDFT",
    orgao_julgador="1ª Vara de Violência Doméstica",
    classe_tpu="12345",
    assuntos_tpu=("Violência Doméstica",),
    data_ajuizamento=date(2024, 5, 10),
    movimentos=(Movement(code="1", name="Conclusos", occurred_at=date(2024, 5, 11)),),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="raw/datajud/2026-11-24/probe.json",
  )


def test_case_is_frozen() -> None:
  with pytest.raises(FrozenInstanceError):
    _case().id = "other"  # type: ignore[misc]


def test_text_id_deterministic() -> None:
  first = text_document_id(TextSource.STATUTE, "Lei 12.318/2010, art. 2º")
  second = text_document_id(TextSource.STATUTE, "Lei 12.318/2010, art. 2º")
  other = text_document_id(TextSource.STATUTE, "Lei 12.318/2010, art. 4º")
  assert first == second
  assert first != other


def test_text_document_fields() -> None:
  doc = TextDocument(
    id=text_document_id(TextSource.DJEN, "Acórdão 1, DJEN 2024-05-10"),
    source=TextSource.DJEN,
    origin_ref="Acórdão 1, DJEN 2024-05-10",
    published_at=date(2024, 5, 10),
    text="trecho sintético",
    citations=("art. 22",),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="raw/djen/x.json",
  )
  assert doc.source is TextSource.DJEN
  assert doc.label_source is LabelSource.PENDING
