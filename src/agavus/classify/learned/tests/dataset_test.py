"""Integration tests for learned-dataset privacy and snapshot boundaries."""

from dataclasses import replace
from datetime import date
from pathlib import Path

from agavus.classify.learned.dataset import prepare_dataset
from agavus.corpus.records import CaseRecord, LabelSource, TextDocument, TextSource
from agavus.corpus.storage import CorpusStorage
from agavus.taxonomy.areas import AREAS
from agavus.taxonomy.labeling import LabelDecision

NEGATIVE_CASES = 100


def _document(label: str, index: int) -> TextDocument:
  return TextDocument(
    id=f"resolved-{label}-{index}",
    source=TextSource.PORTAL,
    origin_ref=f"Acórdão 0700000-{index:02d}.2020.8.07.000{tuple(AREAS).index(label) + 1}",
    published_at=date(2020, 1, 1),
    text=f"texto público {label} {index}",
    citations=(),
    domain_label=label,
    label_source=LabelSource.TPU,
    raw_ref=f"raw/portal/{label}-{index}.json",
  )


def _criminal_document(index: int) -> TextDocument:
  return TextDocument(
    id=f"O-{index}",
    source=TextSource.PORTAL,
    origin_ref=f"Vara Criminal, processo 0700000-{index:02d}.2020.8.07.0099",
    published_at=date(2020, 1, 1),
    text=f"decisão pública criminal {index}",
    citations=(),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref=f"raw/portal/O-{index}.json",
  )


def _reviewed_case() -> CaseRecord:
  return CaseRecord(
    id="0700000-01.2020.8.07.0099",
    tribunal="TJDFT",
    orgao_julgador="Vara Criminal",
    classe_tpu="123",
    assuntos_tpu=(),
    data_ajuizamento=date(2020, 1, 1),
    movimentos=(),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="raw/datajud/review.json",
  )


def _resolved_area_documents(storage: CorpusStorage) -> list[LabelDecision]:
  selected = [_document(label, index) for label in AREAS for index in range(20)]
  storage.save_texts(selected)
  decisions = [_resolved(doc) for doc in selected]
  storage.save_labels(decisions)
  return decisions


def _resolved(document: TextDocument) -> LabelDecision:
  return LabelDecision(document.id, document.domain_label, LabelSource.TPU, False, "resolved")


def _no_additional_negatives(storage: CorpusStorage) -> None:
  result = prepare_dataset(storage, negative_cases=0)
  assert result.report["negative_candidates"] == 0


def test_dataset_excludes_review_private_and_invalid_documents(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  decisions = _resolved_area_documents(storage)
  reviewed = _document("F", 99)
  storage.save_texts(
    [
      reviewed,
      replace(_document("M", 99), id="private", text="CPF 123.456.789-00"),
      replace(_document("F", 98), id="invalid", origin_ref=""),
    ]
  )
  decisions += [LabelDecision(reviewed.id, "F", LabelSource.TPU, True, "review")]
  decisions += [LabelDecision("private", "M", LabelSource.TPU, False, "resolved")]
  decisions += [LabelDecision("invalid", "F", LabelSource.TPU, False, "resolved")]
  storage.save_labels(decisions)
  storage.save_texts([_criminal_document(index) for index in range(NEGATIVE_CASES)])
  dataset = prepare_dataset(storage, negative_cases=NEGATIVE_CASES)
  identifiers = {sample.id for sample in dataset.samples}
  assert not {reviewed.id, "private", "invalid"} & identifiers
  assert (
    sum(
      sample.label == "O" and "snapshot-derived" in sample.label_source
      for sample in dataset.samples
    )
    == NEGATIVE_CASES
  )
  assert dataset.report["excluded"] == {
    "review": 1,
    "privacy": 1,
    "missing_case": 1,
    "unlabeled": 0,
  }
  assert storage.load_labels() == sorted(decisions, key=lambda decision: decision.record_id)


def test_reviewed_sibling_excludes_its_case_from_derived_negatives(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  _resolved_area_documents(storage)
  candidate = _criminal_document(1)
  sibling = replace(candidate, id="reviewed-sibling", text="texto sem rótulo resolvido")
  storage.save_texts([candidate, sibling])
  storage.save_labels([LabelDecision(sibling.id, None, LabelSource.PENDING, True, "review")])
  _no_additional_negatives(storage)


def test_reviewed_case_record_excludes_its_derived_negative_document(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  _resolved_area_documents(storage)
  case = _reviewed_case()
  storage.save_case(case)
  storage.save_text(_criminal_document(1))
  storage.save_labels([LabelDecision(case.id, None, LabelSource.PENDING, True, "review")])
  _no_additional_negatives(storage)
