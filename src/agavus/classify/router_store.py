"""Router persistence in models.db, refitted when its fingerprint changes."""

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from agavus.classify.model import LogRegModel
from agavus.classify.router import QueryRouter, RouterWeights, fingerprint, fit_router
from agavus.retrieve.encoder import Encoder

_SCHEMA = """
CREATE TABLE IF NOT EXISTS routers (
  fingerprint TEXT PRIMARY KEY, classes TEXT NOT NULL, weights BLOB NOT NULL,
  bias BLOB NOT NULL, center BLOB NOT NULL, scale REAL NOT NULL, temperature REAL NOT NULL
)
"""


def _save(conn: sqlite3.Connection, fitted: RouterWeights) -> None:
  with conn:
    conn.execute("DELETE FROM routers")
    conn.execute(
      "INSERT INTO routers VALUES (?,?,?,?,?,?,?)",
      (
        fingerprint(),
        ",".join(fitted.model.classes),
        fitted.model.weights.tobytes(),
        fitted.model.bias.tobytes(),
        fitted.center.tobytes(),
        fitted.scale,
        fitted.temperature,
      ),
    )


def _load(conn: sqlite3.Connection) -> RouterWeights | None:
  row = conn.execute(
    "SELECT classes, weights, bias, center, scale, temperature FROM routers WHERE fingerprint = ?",
    (fingerprint(),),
  ).fetchone()
  if row is None:
    return None
  classes = tuple(str(row[0]).split(","))
  matrix = np.frombuffer(row[1], dtype=np.float64).reshape(-1, len(classes))
  model = LogRegModel(classes, matrix, np.frombuffer(row[2], dtype=np.float64))
  center = np.frombuffer(row[3], dtype=np.float64)
  return RouterWeights(model, center, float(row[4]), float(row[5]))


def ensure_router(db_path: Path, encoder: Encoder) -> QueryRouter:
  """Stored router when current, otherwise fit and store it."""
  db_path.parent.mkdir(parents=True, exist_ok=True)
  with closing(sqlite3.connect(db_path)) as conn:
    conn.execute(_SCHEMA)
    fitted = _load(conn)
    if fitted is None:
      fitted = fit_router(encoder)
      _save(conn, fitted)
  return QueryRouter(encoder, fitted)
