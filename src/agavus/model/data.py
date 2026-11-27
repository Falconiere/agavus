"""Lab data: statute articles only, split by article (docs/llm-lab.md §3).

The source filter is an allowlist inside the query, so decisions are never
even read. The database is opened read-only.
"""

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from agavus.corpus.db import DB_NAME
from agavus.corpus.records import TextSource

ALLOWED_SOURCE = TextSource.STATUTE
HOLDOUT_EVERY = 5


@dataclass(frozen=True)
class Split:
  """Articles the model learns from, and articles it is examined on."""

  train: tuple[str, ...]
  holdout: tuple[str, ...]


def load_articles(data_dir: Path) -> tuple[str, ...]:
  """Statute article texts in stable id order."""
  path = (data_dir / "processed" / DB_NAME).resolve()
  with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
    rows = conn.execute(
      "SELECT text FROM texts WHERE source = ? ORDER BY id", (ALLOWED_SOURCE.value,)
    ).fetchall()
  return tuple(str(row[0]) for row in rows)


def split_articles(articles: Sequence[str]) -> Split:
  """Hold out every 5th article whole, so no article is on both sides."""
  return Split(
    train=tuple(text for index, text in enumerate(articles) if index % HOLDOUT_EVERY),
    holdout=tuple(text for index, text in enumerate(articles) if not index % HOLDOUT_EVERY),
  )
