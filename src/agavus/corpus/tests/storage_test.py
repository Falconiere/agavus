"""Tests for corpus storage (tests folder, tmp dirs, synthetic records)."""

from datetime import date
from pathlib import Path

from agavus.corpus.records import CaseRecord, LabelSource, TextDocument, TextSource
from agavus.corpus.storage import CorpusStorage


def _case(process: str = "0701234-56.2024.8.07.0001") -> CaseRecord:
  return CaseRecord(
    id=process,
    tribunal="TJDFT",
    orgao_julgador="Vara de Família",
    classe_tpu="123",
    assuntos_tpu=("Guarda",),
    data_ajuizamento=date(2024, 1, 2),
    movimentos=(),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="raw/datajud/x.json",
  )


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
  assert storage.save_case(_case()) is True
  assert storage.save_case(_case()) is False
  assert len(list((tmp_path / "processed" / "cases").glob("*.json"))) == 1


def test_save_text_idempotent(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  assert storage.save_text(_text()) is True
  assert storage.save_text(_text()) is False


def test_raw_preserved_with_meta(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  ref = storage.save_raw("datajud", date(2026, 11, 24), "p1.json", b'{"hits": []}', {"size": 1})
  assert (tmp_path / ref).read_bytes() == b'{"hits": []}'
  assert (tmp_path / f"{ref}.meta.json").exists()


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
