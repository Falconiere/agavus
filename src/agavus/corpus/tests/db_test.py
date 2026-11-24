"""Tests for the SQLite store (co-located, tmp dirs, synthetic records)."""

from contextlib import closing
from datetime import date
from pathlib import Path

from agavus.corpus import db
from agavus.corpus.httpfixtures import sample_case
from agavus.corpus.records import LabelSource, TextDocument, TextSource


def test_schema_version(tmp_path: Path) -> None:
  with closing(db.connect(tmp_path)) as conn:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == db.SCHEMA_VERSION


def test_insert_dedup_atomic(tmp_path: Path) -> None:
  with closing(db.connect(tmp_path)) as conn:
    assert db.insert_case(conn, sample_case("p1")) is True
    assert db.insert_case(conn, sample_case("p1")) is False
    assert len(db.select_cases(conn)) == 1


def test_text_roundtrip_preserves_fields(tmp_path: Path) -> None:
  doc = TextDocument(
    id="t1",
    source=TextSource.STATUTE,
    origin_ref="Lei 12.318/2010, art. 2",
    published_at=date(2010, 8, 26),
    text="trecho sintético",
    citations=("art. 2", "Lei 8.069/1990"),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="raw/x.html",
  )
  with closing(db.connect(tmp_path)) as conn:
    assert db.insert_text(conn, doc) is True
    assert db.select_texts(conn) == [doc]


def test_watermarks_and_runs(tmp_path: Path) -> None:
  expected_new = 7
  entry = db.RunEntry(
    run_at="2026-11-24T00:00:00Z",
    source="datajud",
    fetched=10,
    new=expected_new,
    window_start="2020-01-01",
    window_end="2026-11-24",
  )
  with closing(db.connect(tmp_path)) as conn:
    assert db.get_watermark(conn, "datajud") is None
    db.set_watermark(conn, "datajud", "2024-05-10")
    assert db.get_watermark(conn, "datajud") == "2024-05-10"
    db.insert_run(conn, entry)
    runs = db.select_runs(conn)
    assert len(runs) == 1
    assert runs[0]["new"] == expected_new
