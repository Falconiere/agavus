"""Tests for the SQLite store (co-located, tmp dirs, synthetic records)."""

from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np

from agavus.corpus import db
from agavus.corpus.httpfixtures import sample_case, sample_text
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


def test_fts_bm25_search(tmp_path: Path) -> None:
  doc = sample_text(text="medida protetiva de urgência para mulher em situação de violência")
  with closing(db.connect(tmp_path)) as conn:
    assert db.insert_text(conn, doc) is True
    hits = db.search_fts(conn, "protetiva", 10, None)
    assert hits and hits[0][0] == "t1"
    assert db.insert_text(conn, doc) is False
    assert len(db.search_fts(conn, "protetiva", 10, None)) == 1


def test_embeddings_knn(tmp_path: Path) -> None:
  with closing(db.connect(tmp_path)) as conn:
    db.insert_embedding(conn, "a", "v1", np.array([1.0, 0.0]))
    db.insert_embedding(conn, "b", "v1", np.array([0.0, 1.0]))
    db.insert_embedding(conn, "a", "v2", np.array([0.5, 0.5]))
    hits = db.knn_embeddings(conn, np.array([1.0, 0.1]), "v1", k=2)
    assert hits[0][0] == "a"
    assert hits[0][1] > hits[1][1]
    assert len(db.knn_embeddings(conn, np.array([1.0, 0.0]), "v2")) == 1


def test_rebuild_fts_backfills(tmp_path: Path) -> None:
  with closing(db.connect(tmp_path)) as conn:
    conn.execute(
      "INSERT INTO texts VALUES (?,?,?,?,?,?,?,?,?)",
      ("t9", "statute", "ref", None, "guarda compartilhada", "[]", None, "pending", "r"),
    )
    conn.commit()
    assert db.search_fts(conn, "guarda", 10, None) == []
    assert db.rebuild_fts(conn) == 1
    assert db.search_fts(conn, "guarda", 10, None)[0][0] == "t9"


def test_origin_ref_weighted(tmp_path: Path) -> None:
  with closing(db.connect(tmp_path)) as conn:
    db.insert_text(conn, sample_text(doc_id="a22", origin_ref="Lei X, art. 22", text="texto comum"))
    db.insert_text(
      conn, sample_text(doc_id="a10", origin_ref="Lei X, art. 10", text="texto comum 22 22 22")
    )
    hits = db.search_fts(conn, "22", 10, None)
    assert hits[0][0] == "a22"


def test_point_and_batch_text_lookups(tmp_path: Path) -> None:
  expected_new = 2
  d1 = sample_text(doc_id="a1", origin_ref="Lei X, art. 1", text="primeiro")
  d2 = sample_text(doc_id="a2", origin_ref="Lei X, art. 2", text="segundo")
  with closing(db.connect(tmp_path)) as conn:
    assert db.insert_texts(conn, [d1, d2, d1]) == expected_new
    assert db.select_texts_by_ids(conn, []) == {}
    assert db.select_texts_by_ids(conn, ["a2"]) == {"a2": d2}
    assert db.select_text_by_origin_ref(conn, "Lei X, art. 1") == d1
    assert db.select_text_by_origin_ref(conn, "Lei X, art. 99") is None
