"""SQLite artifact store for trained models — no JSON/npz files.

Vocabulary, classes, weights (BLOBs), temperature, and training reports all
live in one models.db. Artifacts are rebuildable by retraining; never in git.
"""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from agavus.classify.features import TfidfVectorizer, VectorizerConfig
from agavus.classify.model import LogRegModel
from agavus.classify.pipeline import TrainedClassifier, TrainReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
  name TEXT PRIMARY KEY, temperature REAL NOT NULL,
  n_features INTEGER NOT NULL, n_classes INTEGER NOT NULL,
  min_df INTEGER NOT NULL, max_features INTEGER NOT NULL,
  char_sizes TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_classes (
  name TEXT NOT NULL, idx INTEGER NOT NULL, label TEXT NOT NULL,
  PRIMARY KEY (name, idx)
);
CREATE TABLE IF NOT EXISTS model_vocab (
  name TEXT NOT NULL, idx INTEGER NOT NULL, term TEXT NOT NULL, idf REAL NOT NULL,
  PRIMARY KEY (name, idx)
);
CREATE TABLE IF NOT EXISTS model_weights (
  name TEXT PRIMARY KEY, weights BLOB NOT NULL, bias BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS train_reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, run_at TEXT NOT NULL,
  holdout_accuracy REAL NOT NULL, holdout_macro_f1 REAL NOT NULL,
  holdout_ece REAL NOT NULL, temperature REAL NOT NULL,
  n_train INTEGER NOT NULL, n_holdout INTEGER NOT NULL,
  n_calibration INTEGER NOT NULL DEFAULT 0
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
  db_path.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(db_path)
  conn.row_factory = sqlite3.Row
  _ = conn.executescript(_SCHEMA)
  columns = {row["name"] for row in conn.execute("PRAGMA table_info(train_reports)")}
  if "n_calibration" not in columns:
    conn.execute("ALTER TABLE train_reports ADD COLUMN n_calibration INTEGER NOT NULL DEFAULT 0")
  return conn


def _save_report(conn: sqlite3.Connection, name: str, report: TrainReport) -> None:
  """Append split counts without erasing reports from older model versions."""
  conn.execute(
    "INSERT INTO train_reports (name, run_at, holdout_accuracy, holdout_macro_f1,"
    " holdout_ece, temperature, n_train, n_holdout, n_calibration) VALUES (?,?,?,?,?,?,?,?,?)",
    (
      name,
      datetime.now(UTC).isoformat(),
      report.holdout_accuracy,
      report.holdout_macro_f1,
      report.holdout_ece,
      report.temperature,
      report.n_train,
      report.n_holdout,
      report.n_calibration,
    ),
  )


def save_model(
  db_path: Path, name: str, classifier: TrainedClassifier, report: TrainReport | None = None
) -> None:
  """Persist (replacing) a named model bundle and optional training report."""
  config = classifier.vectorizer.config
  weights = classifier.model.weights
  with closing(_connect(db_path)) as conn:
    conn.execute("DELETE FROM model_classes WHERE name = ?", (name,))
    conn.execute("DELETE FROM model_vocab WHERE name = ?", (name,))
    conn.execute(
      "INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?,?,?)",
      (
        name,
        classifier.temperature,
        weights.shape[0],
        weights.shape[1],
        config.min_df,
        config.max_features,
        ",".join(str(size) for size in config.char_sizes),
        datetime.now(UTC).isoformat(),
      ),
    )
    conn.executemany(
      "INSERT INTO model_classes VALUES (?,?,?)",
      [(name, index, label) for index, label in enumerate(classifier.model.classes)],
    )
    conn.executemany(
      "INSERT INTO model_vocab VALUES (?,?,?,?)",
      [
        (name, index, term, classifier.vectorizer.idf[index])
        for term, index in classifier.vectorizer.vocab.items()
      ],
    )
    conn.execute(
      "INSERT OR REPLACE INTO model_weights VALUES (?,?,?)",
      (name, weights.astype(np.float64).tobytes(), classifier.model.bias.tobytes()),
    )
    if report is not None:
      _save_report(conn, name, report)
    conn.commit()


def load_model(db_path: Path, name: str) -> TrainedClassifier:
  """Reload a named bundle; raises KeyError when absent."""
  with closing(_connect(db_path)) as conn:
    row = conn.execute("SELECT * FROM models WHERE name = ?", (name,)).fetchone()
    if row is None:
      raise KeyError(f"model not found: {name}")
    classes = tuple(
      str(item["label"])
      for item in conn.execute(
        "SELECT label FROM model_classes WHERE name = ? ORDER BY idx", (name,)
      )
    )
    vocab_rows = conn.execute(
      "SELECT idx, term, idf FROM model_vocab WHERE name = ? ORDER BY idx", (name,)
    ).fetchall()
    weights_row = conn.execute(
      "SELECT weights, bias FROM model_weights WHERE name = ?", (name,)
    ).fetchone()
  config = VectorizerConfig(
    min_df=int(row["min_df"]),
    max_features=int(row["max_features"]),
    char_sizes=tuple(int(part) for part in str(row["char_sizes"]).split(",") if part),
  )
  vectorizer = TfidfVectorizer(config)
  vectorizer.vocab = {str(item["term"]): int(item["idx"]) for item in vocab_rows}
  vectorizer.idf = [float(item["idf"]) for item in vocab_rows]
  n_features = int(row["n_features"])
  n_classes = int(row["n_classes"])
  weights = np.frombuffer(weights_row["weights"], dtype=np.float64).reshape(n_features, n_classes)
  bias = np.frombuffer(weights_row["bias"], dtype=np.float64).reshape(n_classes)
  model = LogRegModel(classes=classes, weights=weights, bias=bias)
  return TrainedClassifier(vectorizer, model, float(row["temperature"]))
