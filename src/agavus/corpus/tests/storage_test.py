"""Tests for corpus storage (tests folder, tmp dirs, synthetic records)."""

from contextlib import closing
from datetime import date
from pathlib import Path

from agavus.corpus import db
from agavus.corpus.httpfixtures import sample_case
from agavus.corpus.records import LabelSource, TextDocument, TextSource
from agavus.corpus.storage import CorpusStorage
from agavus.taxonomy.labeling import LabelDecision


def _text() -> TextDocument:
  return TextDocument(
    id="abc123",
    source=TextSource.STATUTE,
    origin_ref="Lei 12.318/2010, art. 2º",
    published_at=None,
    text="trecho sintético",
    citations=("art. 2º",),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="raw/statute/x.html",
  )


def test_save_case_idempotent(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  assert storage.save_case(sample_case()) is True
  assert storage.save_case(sample_case()) is False
  assert len(storage.load_cases()) == 1


def test_save_text_idempotent(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  assert storage.save_text(_text()) is True
  assert storage.save_text(_text()) is False


def test_raw_preserved_with_meta(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  ref = storage.save_raw(
    "datajud",
    date(2026, 11, 24),
    "p1.json",
    b'{"hits": []}',
    {"url": "https://x.test", "method": "POST", "payload": {"size": 1}},
  )
  assert (tmp_path / ref).read_bytes() == b'{"hits": []}'
  with closing(db.connect(tmp_path)) as conn:
    row = conn.execute("SELECT * FROM raw_payloads WHERE ref = ?", (ref,)).fetchone()
  assert row["url"] == "https://x.test"
  assert row["method"] == "POST"


def test_watermark_roundtrip(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  assert storage.watermark("datajud") is None
  storage.set_watermark("datajud", "2024-05-10")
  assert storage.watermark("datajud") == "2024-05-10"


def test_record_run_appends(tmp_path: Path) -> None:
  expected_runs = 2
  expected_new = 7
  storage = CorpusStorage(tmp_path)
  storage.record_run("datajud", fetched=10, new=7, start=date(2020, 1, 1), end=date(2026, 11, 24))
  storage.record_run("datajud", fetched=3, new=0, start=date(2020, 1, 1), end=date(2026, 11, 24))
  manifest = storage.load_manifest()
  assert len(manifest["runs"]) == expected_runs
  assert manifest["runs"][0]["new"] == expected_new


def test_case_roundtrip(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  original = sample_case(orgao="Vara de Família", assuntos=("Guarda",))
  storage.save_case(original)
  loaded = storage.load_cases()
  assert loaded == [original]


def test_text_roundtrip(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  original = _text()
  storage.save_text(original)
  assert storage.load_texts() == [original]


def test_save_labels_and_review_queue(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  decisions = [
    LabelDecision("a", "M", LabelSource.ORGAN, False, "organ weak label"),
    LabelDecision("b", None, LabelSource.PENDING, True, "no signal"),
  ]
  storage.save_labels(decisions)
  queue = storage.review_queue()
  assert [d.record_id for d in queue] == ["b"]
  storage.save_labels(decisions)
  assert len(storage.review_queue()) == 1


def test_load_empty(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  assert storage.load_cases() == []
  assert storage.load_texts() == []


def test_batch_save_and_targeted_text_load(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  doc = _text()
  assert storage.save_texts([doc, doc]) == 1
  assert storage.load_texts_by_ids(["abc123"]) == {"abc123": doc}
  assert storage.load_text_by_origin_ref(doc.origin_ref) == doc
  assert storage.load_text_by_origin_ref("inexistente") is None
  case = sample_case()
  assert storage.save_cases([case, case]) == 1
