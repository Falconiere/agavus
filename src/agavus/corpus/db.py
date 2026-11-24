"""SQLite store for processed corpus records (docs/corpus/storage.md).

Zero-dep (stdlib), single-writer friendly, indexed dedup, real queries.
Records are immutable once ingested; labels are a derived, re-runnable table.
Raw payloads stay as immutable files — this db is a rebuildable cache.
"""

import json
import sqlite3
from dataclasses import asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from agavus.corpus.records import (
  CaseRecord,
  LabelSource,
  TextDocument,
  case_from_dict,
  text_from_dict,
)
from agavus.corpus.statutes import popular_name
from agavus.taxonomy.labeling import LabelDecision

SCHEMA_VERSION = 1
DB_NAME = "corpus.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta_watermarks (
  source TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta_runs (
  run_at TEXT NOT NULL, source TEXT NOT NULL, fetched INTEGER NOT NULL,
  new INTEGER NOT NULL, window_start TEXT NOT NULL, window_end TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY, tribunal TEXT NOT NULL, orgao_julgador TEXT NOT NULL,
  classe_tpu TEXT NOT NULL, assuntos_tpu TEXT NOT NULL, data_ajuizamento TEXT,
  movimentos TEXT NOT NULL, domain_label TEXT, label_source TEXT NOT NULL,
  raw_ref TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_orgao ON cases(orgao_julgador);
CREATE INDEX IF NOT EXISTS idx_cases_filing ON cases(data_ajuizamento);
CREATE TABLE IF NOT EXISTS texts (
  id TEXT PRIMARY KEY, source TEXT NOT NULL, origin_ref TEXT NOT NULL,
  published_at TEXT, text TEXT NOT NULL, citations TEXT NOT NULL,
  domain_label TEXT, label_source TEXT NOT NULL, raw_ref TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
  record_id TEXT PRIMARY KEY, area TEXT, source TEXT NOT NULL,
  needs_review INTEGER NOT NULL, reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_labels_review ON labels(needs_review, area);
CREATE TABLE IF NOT EXISTS raw_payloads (
  ref TEXT PRIMARY KEY, source TEXT NOT NULL, run TEXT NOT NULL,
  url TEXT, method TEXT, payload_json TEXT, fetched_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS texts_fts
  USING fts5(doc_id UNINDEXED, origin_ref, law_name, text);
CREATE TABLE IF NOT EXISTS embeddings (
  doc_id TEXT NOT NULL, model_version TEXT NOT NULL, dim INTEGER NOT NULL,
  vector BLOB NOT NULL, PRIMARY KEY (doc_id, model_version)
);
"""


def json_default(value: object) -> str:
  """Serialize dates/enums inside JSON columns and raw metadata."""
  if isinstance(value, date | datetime):
    return value.isoformat()
  if isinstance(value, Enum):
    return str(value.value)
  raise TypeError(f"not JSON serializable: {type(value)!r}")


def connect(root: Path) -> sqlite3.Connection:
  """Open (creating if needed) the corpus database with schema applied."""
  path = root / "processed" / DB_NAME
  path.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(path)
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA journal_mode=WAL")
  conn.executescript(_SCHEMA)
  _migrate_fts(conn)
  conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
  return conn


def _migrate_fts(conn: sqlite3.Connection) -> None:
  """FTS index is rebuildable: recreate when the column layout changed."""
  columns = {row[1] for row in conn.execute("PRAGMA table_info(texts_fts)")}
  if "law_name" not in columns:
    conn.execute("DROP TABLE texts_fts")
    conn.execute(
      "CREATE VIRTUAL TABLE texts_fts USING fts5(doc_id UNINDEXED, origin_ref, law_name, text)"
    )
    conn.commit()


def _insert(conn: sqlite3.Connection, table: str, values: tuple[object, ...]) -> bool:
  placeholders = ",".join("?" for _ in values)
  cursor = conn.execute(f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})", values)
  conn.commit()
  return cursor.rowcount > 0


def insert_case(conn: sqlite3.Connection, record: CaseRecord) -> bool:
  """Atomic dedup insert; False when the id already existed."""
  return _insert(
    conn,
    "cases",
    (
      record.id,
      record.tribunal,
      record.orgao_julgador,
      record.classe_tpu,
      json.dumps(record.assuntos_tpu, ensure_ascii=False),
      record.data_ajuizamento.isoformat() if record.data_ajuizamento else None,
      json.dumps([asdict(m) for m in record.movimentos], default=json_default),
      record.domain_label,
      record.label_source.value,
      record.raw_ref,
    ),
  )


def insert_text(conn: sqlite3.Connection, record: TextDocument) -> bool:
  """Atomic dedup insert; False when the id already existed. Indexes FTS."""
  inserted = _insert(
    conn,
    "texts",
    (
      record.id,
      record.source.value,
      record.origin_ref,
      record.published_at.isoformat() if record.published_at else None,
      record.text,
      json.dumps(record.citations, ensure_ascii=False),
      record.domain_label,
      record.label_source.value,
      record.raw_ref,
    ),
  )
  if inserted:
    conn.execute(
      "INSERT INTO texts_fts (doc_id, origin_ref, law_name, text) VALUES (?,?,?,?)",
      (record.id, record.origin_ref, popular_name(record.origin_ref), record.text),
    )
    conn.commit()
  return inserted


def search_fts(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[tuple[str, float]]:
  """BM25-ranked ids; origin_ref 10x, law_name 8x (names/numbers decisive)."""
  rows = conn.execute(
    "SELECT doc_id, bm25(texts_fts, 0.0, 10.0, 8.0, 1.0) AS score FROM texts_fts "
    "WHERE texts_fts MATCH ? ORDER BY score LIMIT ?",
    (query, limit),
  ).fetchall()
  return [(str(row["doc_id"]), float(row["score"])) for row in rows]


def rebuild_fts(conn: sqlite3.Connection) -> int:
  """Rebuild the FTS index from the texts table (idempotent maintenance)."""
  conn.execute("DELETE FROM texts_fts")
  rows = conn.execute("SELECT id, origin_ref, text FROM texts").fetchall()
  conn.executemany(
    "INSERT INTO texts_fts (doc_id, origin_ref, law_name, text) VALUES (?,?,?,?)",
    [
      (
        str(row["id"]),
        str(row["origin_ref"]),
        popular_name(str(row["origin_ref"])),
        str(row["text"]),
      )
      for row in rows
    ],
  )
  conn.commit()
  return len(rows)


class RawMeta(TypedDict):
  """Audit metadata for one preserved raw payload."""

  ref: str
  source: str
  run: str
  url: str | None
  method: str | None
  payload_json: str | None
  fetched_at: str


def insert_raw_meta(conn: sqlite3.Connection, meta: RawMeta) -> None:
  """Register a preserved raw payload in the audit table (no JSON sidecars)."""
  conn.execute(
    "INSERT OR REPLACE INTO raw_payloads VALUES (?,?,?,?,?,?,?)",
    (
      meta["ref"],
      meta["source"],
      meta["run"],
      meta["url"],
      meta["method"],
      meta["payload_json"],
      meta["fetched_at"],
    ),
  )
  conn.commit()


def insert_embedding(
  conn: sqlite3.Connection, doc_id: str, model_version: str, vector: NDArray[np.float64]
) -> None:
  """Store a dense vector as float32 BLOB, keyed by doc + model version."""
  vec = np.asarray(vector, dtype=np.float32)
  conn.execute(
    "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
    (doc_id, model_version, vec.size, vec.tobytes()),
  )
  conn.commit()


def knn_embeddings(
  conn: sqlite3.Connection,
  query: NDArray[np.float64],
  model_version: str,
  k: int = 5,
) -> list[tuple[str, float]]:
  """Brute-force cosine KNN (fine at corpus scale; sqlite-vec later if needed)."""
  rows = conn.execute(
    "SELECT doc_id, vector FROM embeddings WHERE model_version = ?", (model_version,)
  ).fetchall()
  query_vec = np.asarray(query, dtype=np.float32)
  query_norm = float(np.linalg.norm(query_vec))
  scored: list[tuple[str, float]] = []
  for row in rows:
    vec = np.frombuffer(row["vector"], dtype=np.float32)
    denom = query_norm * float(np.linalg.norm(vec))
    score = float(np.dot(query_vec, vec) / denom) if denom else 0.0
    scored.append((str(row["doc_id"]), score))
  scored.sort(key=lambda pair: -pair[1])
  return scored[:k]


def _row_dict(row: sqlite3.Row, json_fields: tuple[str, ...]) -> dict[str, object]:
  keys = tuple(row.keys())
  data: dict[str, object] = {key: row[key] for key in keys}
  for field in json_fields:
    data[field] = json.loads(str(row[field]))
  return data


def select_cases(conn: sqlite3.Connection) -> list[CaseRecord]:
  """All case records in stable id order."""
  rows = conn.execute("SELECT * FROM cases ORDER BY id").fetchall()
  return [case_from_dict(_row_dict(row, ("assuntos_tpu", "movimentos"))) for row in rows]


def select_texts(conn: sqlite3.Connection) -> list[TextDocument]:
  """All text documents in stable id order."""
  rows = conn.execute("SELECT * FROM texts ORDER BY id").fetchall()
  return [text_from_dict(_row_dict(row, ("citations",))) for row in rows]


def get_watermark(conn: sqlite3.Connection, source: str) -> str | None:
  """Stored incremental position for a source, if any."""
  row = conn.execute("SELECT value FROM meta_watermarks WHERE source = ?", (source,)).fetchone()
  return str(row["value"]) if row is not None else None


def set_watermark(conn: sqlite3.Connection, source: str, value: str) -> None:
  """Advance a source watermark (only after a successful run)."""
  conn.execute("INSERT OR REPLACE INTO meta_watermarks VALUES (?,?)", (source, value))
  conn.commit()


class RunEntry(TypedDict):
  """One ingestion run, recorded for auditability."""

  run_at: str
  source: str
  fetched: int
  new: int
  window_start: str
  window_end: str


def insert_run(conn: sqlite3.Connection, entry: RunEntry) -> None:
  """Append one audit-trail run entry."""
  conn.execute(
    "INSERT INTO meta_runs VALUES (?,?,?,?,?,?)",
    (
      entry["run_at"],
      entry["source"],
      entry["fetched"],
      entry["new"],
      entry["window_start"],
      entry["window_end"],
    ),
  )
  conn.commit()


def select_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
  """All run entries in insertion order."""
  return conn.execute("SELECT * FROM meta_runs ORDER BY rowid").fetchall()


def select_watermarks(conn: sqlite3.Connection) -> dict[str, str]:
  """All source watermarks."""
  rows = conn.execute("SELECT source, value FROM meta_watermarks").fetchall()
  return {str(row["source"]): str(row["value"]) for row in rows}


def upsert_labels(conn: sqlite3.Connection, decisions: list[LabelDecision]) -> None:
  """Replace derived labels (re-runnable labeling pass)."""
  conn.executemany(
    "INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?)",
    [
      (d.record_id, d.domain_label, d.label_source.value, int(d.needs_review), d.reason)
      for d in decisions
    ],
  )
  conn.commit()


def select_labels(conn: sqlite3.Connection, *, review_only: bool = False) -> list[LabelDecision]:
  """Label decisions, optionally filtered to the review queue."""
  query = "SELECT * FROM labels"
  if review_only:
    query += " WHERE needs_review = 1"
  rows = conn.execute(query + " ORDER BY record_id").fetchall()
  return [
    LabelDecision(
      record_id=str(row["record_id"]),
      domain_label=str(row["area"]) if row["area"] is not None else None,
      label_source=LabelSource(str(row["source"])),
      needs_review=bool(row["needs_review"]),
      reason=str(row["reason"]),
    )
    for row in rows
  ]
