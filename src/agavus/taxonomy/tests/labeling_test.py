"""Tests for the labeling pass (co-located, synthetic records)."""

from datetime import date

import pytest

from agavus.corpus.records import CaseRecord, LabelSource, TextDocument, TextSource
from agavus.taxonomy import tpu
from agavus.taxonomy.labeling import label_case, label_text


def _case(orgao: str = "Vara Cível", assuntos: tuple[str, ...] = ()) -> CaseRecord:
  return CaseRecord(
    id="0701234-56.2024.8.07.0001",
    tribunal="TJDFT",
    orgao_julgador=orgao,
    classe_tpu="123",
    assuntos_tpu=assuntos,
    data_ajuizamento=date(2024, 1, 2),
    movimentos=(),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="raw/x.json",
  )


def test_organ_weak_label() -> None:
  decision = label_case(_case(orgao="1ª Vara de Violência Doméstica"))
  assert decision.domain_label == "M"
  assert decision.label_source is LabelSource.ORGAN
  assert not decision.needs_review


def test_no_signal_goes_to_review() -> None:
  decision = label_case(_case())
  assert decision.domain_label is None
  assert decision.needs_review


def test_tpu_overrides_organ(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setitem(tpu.CODE_TO_AREA, "111", "F")
  decision = label_case(_case(orgao="Vara de Família", assuntos=("111",)))
  assert decision.domain_label == "F"
  assert decision.label_source is LabelSource.TPU


def test_tpu_organ_conflict_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setitem(tpu.CODE_TO_AREA, "555", "F")
  decision = label_case(_case(orgao="Vara de Violência Doméstica", assuntos=("555",)))
  assert decision.domain_label == "F"
  assert decision.needs_review
  assert "conflict" in decision.reason


def test_mixed_areas_provisional_m(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setitem(tpu.CODE_TO_AREA, "111", "M")
  monkeypatch.setitem(tpu.CODE_TO_AREA, "222", "F")
  decision = label_case(_case(assuntos=("111", "222")))
  assert decision.domain_label == "M"
  assert decision.needs_review


def test_statute_map() -> None:
  doc = TextDocument(
    id="x",
    source=TextSource.STATUTE,
    origin_ref="Lei 12.318/2010, art. 2",
    published_at=None,
    text="trecho",
    citations=(),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="raw/x.html",
  )
  decision = label_text(doc)
  assert decision.domain_label == "F"
  assert decision.label_source is LabelSource.MANUAL
