"""Corpus storage facade: immutable raw files + SQLite processed store.

Layout (docs/corpus/storage.md): data/raw/<source>/<run>/… verbatim payloads;
data/processed/corpus.db holds records, labels, runs, watermarks. Raw is never
mutated; record inserts are atomic-dedup; the whole store is a rebuildable
cache that never enters git.
"""

import json
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from agavus.corpus import db
from agavus.corpus.db import RunEntry
from agavus.corpus.records import CaseRecord, TextDocument
from agavus.taxonomy.labeling import LabelDecision


class Manifest(TypedDict):
  """Audit trail of runs plus per-source incremental watermarks."""

  runs: list[RunEntry]
  watermarks: dict[str, str]


class CorpusStorage:
  """Filesystem + SQLite corpus store rooted at the configured data dir."""

  def __init__(self, root: Path) -> None:
    self.root = root
    self.raw = root / "raw"
    self.processed = root / "processed"

  def save_raw(self, source: str, run: date, name: str, payload: bytes, meta: object) -> str:
    """Preserve a fetched payload verbatim; register metadata in SQLite."""
    folder = self.raw / source / run.isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(payload)
    ref = str((folder / name).relative_to(self.root))
    details = meta if isinstance(meta, dict) else {}
    with closing(db.connect(self.root)) as conn:
      db.insert_raw_meta(
        conn,
        db.RawMeta(
          ref=ref,
          source=source,
          run=run.isoformat(),
          url=str(details.get("url")) if details.get("url") else None,
          method=str(details.get("method")) if details.get("method") else None,
          payload_json=(
            json.dumps(details.get("payload"), ensure_ascii=False)
            if details.get("payload")
            else None
          ),
          fetched_at=datetime.now(UTC).isoformat(),
        ),
      )
    return ref

  def save_case(self, record: CaseRecord) -> bool:
    """Store a case record; False when it already existed (idempotent)."""
    with closing(db.connect(self.root)) as conn:
      return db.insert_case(conn, record)

  def save_text(self, record: TextDocument) -> bool:
    """Store a text document; False when it already existed (idempotent)."""
    with closing(db.connect(self.root)) as conn:
      return db.insert_text(conn, record)

  def load_cases(self) -> list[CaseRecord]:
    """Read all stored case records, in stable order."""
    with closing(db.connect(self.root)) as conn:
      return db.select_cases(conn)

  def load_texts(self) -> list[TextDocument]:
    """Read all stored text documents, in stable order."""
    with closing(db.connect(self.root)) as conn:
      return db.select_texts(conn)

  def load_manifest(self) -> Manifest:
    """Assemble the manifest view from runs + watermarks tables."""
    with closing(db.connect(self.root)) as conn:
      runs = [
        RunEntry(
          run_at=str(row["run_at"]),
          source=str(row["source"]),
          fetched=int(row["fetched"]),
          new=int(row["new"]),
          window_start=str(row["window_start"]),
          window_end=str(row["window_end"]),
        )
        for row in db.select_runs(conn)
      ]
      return Manifest(runs=runs, watermarks=db.select_watermarks(conn))

  def record_run(self, source: str, fetched: int, new: int, start: date, end: date) -> None:
    """Append a run entry to the audit trail."""
    entry = RunEntry(
      run_at=datetime.now(UTC).isoformat(),
      source=source,
      fetched=fetched,
      new=new,
      window_start=start.isoformat(),
      window_end=end.isoformat(),
    )
    with closing(db.connect(self.root)) as conn:
      db.insert_run(conn, entry)

  def watermark(self, source: str) -> str | None:
    """Last successfully ingested position for a source."""
    with closing(db.connect(self.root)) as conn:
      return db.get_watermark(conn, source)

  def set_watermark(self, source: str, value: str) -> None:
    """Advance a source watermark (only after a successful run)."""
    with closing(db.connect(self.root)) as conn:
      db.set_watermark(conn, source, value)

  def save_labels(self, decisions: list[LabelDecision]) -> None:
    """Persist the derived labeling pass (re-runnable)."""
    with closing(db.connect(self.root)) as conn:
      db.upsert_labels(conn, decisions)

  def review_queue(self) -> list[LabelDecision]:
    """Decisions flagged for human review."""
    with closing(db.connect(self.root)) as conn:
      return db.select_labels(conn, review_only=True)

  def load_labels(self) -> list[LabelDecision]:
    """All derived label decisions."""
    with closing(db.connect(self.root)) as conn:
      return db.select_labels(conn)

  def search_texts(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
    """BM25-ranked text ids (FTS5)."""
    with closing(db.connect(self.root)) as conn:
      return db.search_fts(conn, query, limit)

  def rebuild_index(self) -> int:
    """Rebuild the FTS index from stored texts; returns indexed count."""
    with closing(db.connect(self.root)) as conn:
      return db.rebuild_fts(conn)

  def save_embedding(self, doc_id: str, model_version: str, vector: NDArray[np.float64]) -> None:
    """Store a dense vector for a document (tier-4 retrieval)."""
    with closing(db.connect(self.root)) as conn:
      db.insert_embedding(conn, doc_id, model_version, vector)

  def search_similar(
    self, query: NDArray[np.float64], model_version: str, k: int = 5
  ) -> list[tuple[str, float]]:
    """Cosine KNN over stored embeddings of one model version."""
    with closing(db.connect(self.root)) as conn:
      return db.knn_embeddings(conn, query, model_version, k)
