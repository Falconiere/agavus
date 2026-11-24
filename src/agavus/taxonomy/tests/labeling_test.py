"""Tests for the labeling pass (co-located, synthetic records)."""

import pytest

from agavus.corpus.httpfixtures import sample_case
from agavus.corpus.records import LabelSource, TextDocument, TextSource
from agavus.taxonomy import tpu
from agavus.taxonomy.labeling import label_case, label_text


def test_organ_weak_label() -> None:
  decision = label_case(sample_case(orgao="1ª Vara de Violência Doméstica"))
  assert decision.domain_label == "M"
  assert decision.label_source is LabelSource.ORGAN
  assert not decision.needs_review


def test_no_signal_goes_to_review() -> None:
  decision = label_case(sample_case())
  assert decision.domain_label is None
  assert decision.needs_review


def test_tpu_overrides_organ(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setitem(tpu.CODE_TO_AREA, "111", "F")
  decision = label_case(sample_case(orgao="Vara de Família", assuntos=("111",)))
  assert decision.domain_label == "F"
  assert decision.label_source is LabelSource.TPU


def test_tpu_organ_conflict_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setitem(tpu.CODE_TO_AREA, "555", "F")
  decision = label_case(sample_case(orgao="Vara de Violência Doméstica", assuntos=("555",)))
  assert decision.domain_label == "F"
  assert decision.needs_review
  assert "conflict" in decision.reason


def test_mixed_areas_provisional_m(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setitem(tpu.CODE_TO_AREA, "111", "M")
  monkeypatch.setitem(tpu.CODE_TO_AREA, "222", "F")
  decision = label_case(sample_case(assuntos=("111", "222")))
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


def _djen(doc_id: str, origin: str, text: str) -> TextDocument:
  return TextDocument(
    id=doc_id,
    source=TextSource.DJEN,
    origin_ref=origin,
    published_at=None,
    text=text,
    citations=(),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="raw/djen.json",
  )


def test_djen_origin_uses_organ() -> None:
  doc = _djen("djen-1", "Decisão, 1ª Vara de Família, processo 1, DJEN 2024-05-10, id 9", "guarda")
  decision = label_text(doc)
  assert decision.domain_label == "F"
  assert decision.label_source is LabelSource.ORGAN


def test_jury_origin_stays_in_review() -> None:
  doc = _djen("djen-2", "Acórdão, Tribunal do Júri, processo 1, DJEN 2024-05-10, id 8", "trecho")
  decision = label_text(doc)
  assert decision.domain_label is None
  assert decision.needs_review
  assert decision.reason == "júri without tpu"
