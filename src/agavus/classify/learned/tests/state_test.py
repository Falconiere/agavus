"""SQLite state must preserve private provenance and refuse overwriting a run."""

import sqlite3
import zipfile
from pathlib import Path

import pytest

from agavus.classify.learned.dataset import Dataset
from agavus.classify.learned.partitions import Partitions
from agavus.classify.learned.state import ExperimentState
from agavus.classify.learned.types import Sample


def test_snapshot_roundtrip_and_unique_run_directory(tmp_path: Path) -> None:
  sample = Sample(
    "id",
    "fictional text",
    "support",
    "family",
    "synthetic:1",
    "authored",
    statement="fictional statement",
    template="template:1",
  )
  dataset = Dataset((sample,), Partitions((0,), (), (), ()), {"excluded": 0})
  folder = tmp_path / "run"
  state = ExperimentState.create(folder, dataset, {"seed": 0})
  state.put("status", "partial")
  source = tmp_path / "example.py"
  source.write_text("print('synthetic')\n")
  state.capture_source([source], tmp_path)
  with zipfile.ZipFile(folder / "source.zip") as archive:
    assert archive.read("example.py") == source.read_bytes()
  assert state.get("status") == "partial"
  with sqlite3.connect(folder / "experiment.db") as conn:
    assert conn.execute("SELECT partition FROM samples").fetchall() == [("train",)]
    assert "synthetic:1" in str(conn.execute("SELECT payload FROM samples").fetchone())
  with pytest.raises(FileExistsError):
    ExperimentState.create(folder, dataset, {})
