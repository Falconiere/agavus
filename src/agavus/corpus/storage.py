"""Corpus storage: raw preservation, idempotent processed records, manifest.

Layout (docs/corpus/storage.md): data/raw/<source>/<run>/…, data/processed/
cases|texts/<id>.json, manifest.json with per-source watermarks. Raw is never
mutated; re-saving an existing record is a no-op.
"""

import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict

from agavus.corpus.records import CaseRecord, TextDocument

MANIFEST_NAME = "manifest.json"


class RunEntry(TypedDict):
  """One ingestion run, recorded for auditability."""

  run_at: str
  source: str
  fetched: int
  new: int
  window_start: str
  window_end: str


class Manifest(TypedDict):
  """Audit trail of runs plus per-source incremental watermarks."""

  runs: list[RunEntry]
  watermarks: dict[str, str]


def _json_default(value: object) -> str:
  if isinstance(value, date | datetime):
    return value.isoformat()
  if isinstance(value, Enum):
    return str(value.value)
  raise TypeError(f"not JSON serializable: {type(value)!r}")


def _dump(record: CaseRecord | TextDocument) -> str:
  return json.dumps(asdict(record), default=_json_default, ensure_ascii=False, indent=2)


class CorpusStorage:
  """Filesystem-backed corpus store rooted at the configured data dir."""

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
      json.dumps(meta, default=_json_default, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str((folder / name).relative_to(self.root))

  def _save(self, kind: str, record_id: str, payload: str) -> bool:
    folder = self.processed / kind
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{record_id}.json"
    if target.exists():
      return False
    target.write_text(payload, encoding="utf-8")
    return True

  def save_case(self, record: CaseRecord) -> bool:
    """Store a case record; False when it already existed (idempotent)."""
    return self._save("cases", record.id, _dump(record))

  def save_text(self, record: TextDocument) -> bool:
    """Store a text document; False when it already existed (idempotent)."""
    return self._save("texts", record.id, _dump(record))

  def load_manifest(self) -> Manifest:
    """Read the manifest, or an empty one when absent."""
    path = self.processed / MANIFEST_NAME
    if not path.exists():
      return Manifest(runs=[], watermarks={})
    data: Manifest = json.loads(path.read_text(encoding="utf-8"))
    return data

  def save_manifest(self, manifest: Manifest) -> None:
    """Persist the manifest atomically enough for a single-writer pipeline."""
    self.processed.mkdir(parents=True, exist_ok=True)
    path = self.processed / MANIFEST_NAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

  def record_run(self, source: str, fetched: int, new: int, start: date, end: date) -> None:
    """Append a run entry to the manifest."""
    manifest = self.load_manifest()
    manifest["runs"].append(
      RunEntry(
        run_at=datetime.now(UTC).isoformat(),
        source=source,
        fetched=fetched,
        new=new,
        window_start=start.isoformat(),
        window_end=end.isoformat(),
      )
    )
    self.save_manifest(manifest)

  def watermark(self, source: str) -> str | None:
    """Last successfully ingested position for a source."""
    return self.load_manifest()["watermarks"].get(source)

  def set_watermark(self, source: str, value: str) -> None:
    """Advance a source watermark (only after a successful run)."""
    manifest = self.load_manifest()
    manifest["watermarks"][source] = value
    self.save_manifest(manifest)
