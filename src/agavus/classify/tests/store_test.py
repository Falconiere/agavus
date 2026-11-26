"""Tests for the SQLite model artifact store (tests folder)."""

import sqlite3
from pathlib import Path

import pytest

from agavus.classify import store
from agavus.classify.features import VectorizerConfig
from agavus.classify.pipeline import TrainedClassifier, TrainReport, train_classifier

TEXTS = [
  "protetiva violência",
  "protetiva ameaça",
  "protetiva urgência",
  "guarda filhos",
  "divórcio bens",
  "alimentos criança",
]
LABELS = ["M", "M", "M", "F", "F", "F"]


def _trained_classifier() -> tuple[TrainedClassifier, TrainReport]:
  return train_classifier(
    TEXTS, LABELS, vectorizer_config=VectorizerConfig(min_df=1, char_sizes=())
  )


def test_missing_model_raises(tmp_path: Path) -> None:
  with pytest.raises(KeyError):
    store.load_model(tmp_path / "models.db", "classifier")


def test_replace_keeps_single_bundle(tmp_path: Path) -> None:
  db_path = tmp_path / "models.db"
  classifier, report = _trained_classifier()
  store.save_model(db_path, "classifier", classifier, report)
  store.save_model(db_path, "classifier", classifier, report)
  loaded = store.load_model(db_path, "classifier")
  assert loaded.model.classes == classifier.model.classes
  assert len(loaded.vectorizer.vocab) == len(classifier.vectorizer.vocab)


def test_legacy_report_rows_receive_default_calibration_count(tmp_path: Path) -> None:
  """Catch migrations that erase or misrepresent reports written before calibration counts."""
  db_path = tmp_path / "models.db"
  with sqlite3.connect(db_path) as conn:
    conn.execute(
      "CREATE TABLE train_reports (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
      "run_at TEXT NOT NULL, holdout_accuracy REAL NOT NULL, holdout_macro_f1 REAL NOT NULL, "
      "holdout_ece REAL NOT NULL, temperature REAL NOT NULL, n_train INTEGER NOT NULL, "
      "n_holdout INTEGER NOT NULL)"
    )
    conn.execute(
      "INSERT INTO train_reports VALUES (1, 'legacy', '2024-01-01T00:00:00Z', 1, 1, 0, 1, 6, 2)"
    )
  classifier, report = _trained_classifier()

  store.save_model(db_path, "classifier", classifier, report)

  with sqlite3.connect(db_path) as conn:
    rows = conn.execute("SELECT name, n_calibration FROM train_reports ORDER BY id").fetchall()
  assert rows == [("legacy", 0), ("classifier", report.n_calibration)]
