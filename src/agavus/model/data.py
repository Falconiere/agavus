"""Lab data from the corpus: statute articles, split by article, and case text (llm-lab §3).

The source filter is an allowlist inside the query. Case text is the anonymized DJEN text
stored at ingestion, never `data/raw/`. The database is opened read-only.
"""

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from agavus.corpus.anonymize import contains_personal_data
from agavus.corpus.db import DB_NAME
from agavus.corpus.records import TextSource

ALLOWED_SOURCE = TextSource.STATUTE
HOLDOUT_EVERY = 5


@dataclass(frozen=True)
class Split:
  """Articles the model learns from, and articles it is examined on."""

  train: tuple[str, ...]
  holdout: tuple[str, ...]


def _rows(data_dir: Path, source: TextSource) -> tuple[tuple[str, str], ...]:
  """(origin_ref, text) of one source in id order, read-only."""
  path = (data_dir / "processed" / DB_NAME).resolve()
  with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
    query = "SELECT origin_ref, text FROM texts WHERE source = ? ORDER BY id"
    return tuple((str(ref), str(text)) for ref, text in conn.execute(query, (source.value,)))


def _texts(data_dir: Path, source: TextSource) -> tuple[str, ...]:
  return tuple(text for _, text in _rows(data_dir, source))


def load_articles(data_dir: Path) -> tuple[str, ...]:
  """Statute article texts in stable id order."""
  return _texts(data_dir, ALLOWED_SOURCE)


def load_statute_articles(data_dir: Path) -> tuple[tuple[str, str], ...]:
  """(origin_ref, text) of every statute article, e.g. ('Lei 11.340/2006, art. 22', …)."""
  return _rows(data_dir, ALLOWED_SOURCE)


def load_case_texts(data_dir: Path) -> tuple[str, ...]:
  """Anonymized DJEN texts, once each, minus any the residual personal-data scan flags."""
  unique = dict.fromkeys(_texts(data_dir, TextSource.DJEN))
  return tuple(text for text in unique if not contains_personal_data(text))


def split_articles(articles: Sequence[str]) -> Split:
  """Hold out every 5th article whole, so no article is on both sides."""
  return Split(
    train=tuple(text for index, text in enumerate(articles) if index % HOLDOUT_EVERY),
    holdout=tuple(text for index, text in enumerate(articles) if not index % HOLDOUT_EVERY),
  )
