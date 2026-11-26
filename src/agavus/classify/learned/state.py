"""Append-only run directories with SQLite snapshots, parameters and results."""

import hashlib
import json
import sqlite3
import zipfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Self

from agavus.classify.learned.dataset import Dataset
from agavus.corpus.db import json_default
from agavus.corpus.jsonutil import as_dict


class ExperimentState:
  """Private experiment state; creating a run never overwrites an older run."""

  def __init__(self, folder: Path) -> None:
    self.folder = folder
    self.path = folder / "experiment.db"

  @classmethod
  def create(cls, folder: Path, dataset: Dataset, parameters: dict[str, object]) -> Self:
    """Freeze row content, provenance and membership before any model computation."""
    folder.mkdir(parents=True, exist_ok=False)
    state = cls(folder)
    parts = dataset.partitions
    membership = {
      i: name
      for name in ("train", "validation", "calibration", "final")
      for i in getattr(parts, name)
    }
    with sqlite3.connect(state.path) as conn:
      conn.execute("CREATE TABLE state (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
      conn.execute(
        "CREATE TABLE samples (position INTEGER PRIMARY KEY, partition TEXT, payload TEXT)"
      )
      conn.executemany(
        "INSERT INTO samples VALUES (?,?,?)",
        (
          (i, membership[i], json.dumps(asdict(s), default=json_default, ensure_ascii=False))
          for i, s in enumerate(dataset.samples)
        ),
      )
    state.put("parameters", parameters)
    state.put("dataset", dataset.report)
    state.put("status", "prepared")
    return state

  def put(self, key: str, value: object) -> None:
    """Atomically record one result; JSON is a SQLite column, never a state file."""
    with sqlite3.connect(self.path) as conn:
      conn.execute(
        "INSERT OR REPLACE INTO state VALUES (?,?)",
        (key, json.dumps(value, default=json_default, ensure_ascii=False)),
      )

  def get(self, key: str) -> object:
    """Read one recorded value for reports and interrupted-run inspection."""
    with sqlite3.connect(self.path) as conn:
      row = conn.execute("SELECT payload FROM state WHERE key=?", (key,)).fetchone()
    if row is None:
      raise KeyError(key)
    value: object = json.loads(str(row[0]))
    return value

  def results(self) -> dict[str, dict[str, object]]:
    """Keep completed measurements visible if a later evaluation reaches its deadline."""
    with sqlite3.connect(self.path) as conn:
      rows = conn.execute("SELECT key,payload FROM state WHERE key LIKE 'result:%'").fetchall()
    return {
      str(key).removeprefix("result:"): as_dict(json.loads(str(payload))) for key, payload in rows
    }

  def capture_source(self, paths: Sequence[Path], root: Path) -> None:
    """Seal public source, including untracked implementation files, for replay."""
    manifest: dict[str, str] = {}
    with zipfile.ZipFile(self.folder / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
      for path in sorted(paths):
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        payload = path.read_bytes()
        archive.writestr(relative, payload)
        manifest[relative] = hashlib.sha256(payload).hexdigest()
    self.put("source_sha256", manifest)
