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
    """Preserve a fetched payload verbatim plus its query metadata; return ref."""
    folder = self.raw / source / run.isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(payload)
    (folder / f"{name}.meta.json").write_text(
      json.dumps(meta, default=db.json_default, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str((folder / name).relative_to(self.root))

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
