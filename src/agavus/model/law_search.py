"""BM25 search over every Planalto article, so Agathe answers from retrieved law (§9).

The index is SQLite FTS5 with accents folded. Rows keep the verbatim `origin_ref` and the
law's citable name; answers and chat sources cite them as they are.
"""

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from agavus.model.planalto import PLANALTO_DIR
from agavus.model.sft_rules import article_number, load_rule_sources

SEARCH_FILE = PLANALTO_DIR / "search.db"
STOP_WORDS = frozenset(
  {"de", "da", "do", "o", "a", "que", "e", "em", "um", "uma", "para", "com", "por", "no", "na"}
  | {"os", "as", "dos", "das"}
)
# Shorter words are dropped from queries: single letters match nearly every article.
MIN_TERM = 2
WORD = re.compile(r"\w+")
SCHEMA = """
CREATE TABLE articles (
  id INTEGER PRIMARY KEY, origin_ref TEXT, url TEXT, name TEXT, number TEXT, text TEXT
);
CREATE INDEX articles_by_ref ON articles (origin_ref);
CREATE VIRTUAL TABLE search USING fts5 (
  origin_ref UNINDEXED, url UNINDEXED, name, number, text,
  content = 'articles', content_rowid = 'id', tokenize = 'unicode61 remove_diacritics 2'
);
"""


@dataclass(frozen=True)
class Hit:
  """One article found: its citation, the law's name, the article number and its text."""

  origin_ref: str
  name: str
  number: str
  text: str


def build_index(data_dir: Path) -> Path:
  """Index every crawled Planalto article; rebuilt whole, replacing the old index at once."""
  cards, articles = load_rule_sources(data_dir)
  names = {card.url: card.name for card in cards}
  path = data_dir / SEARCH_FILE
  temporary = path.with_name(path.name + ".tmp")
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary.unlink(missing_ok=True)
  with closing(sqlite3.connect(temporary)) as connection, connection:
    connection.executescript(SCHEMA)
    connection.executemany(
      "INSERT INTO articles (origin_ref, url, name, number, text) VALUES (?, ?, ?, ?, ?)",
      (
        (origin_ref, url, names.get(url, url), article_number(origin_ref, text), text)
        for url, found in sorted(articles.items())
        for origin_ref, text in found
      ),
    )
    connection.execute("INSERT INTO search (search) VALUES ('rebuild')")
  temporary.replace(path)
  return path


def _connect(db: Path) -> sqlite3.Connection:
  """Read-only, so a missing index raises instead of becoming an empty file."""
  return sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)


def index_counts(db: Path) -> tuple[int, int]:
  """(articles, laws) in the index."""
  with closing(_connect(db)) as connection:
    articles, laws = connection.execute(
      "SELECT count(*), count(DISTINCT url) FROM articles"
    ).fetchone()
  return int(articles), int(laws)


def match_terms(query: str) -> str:
  """The query as an OR of quoted words, stop words and single letters dropped."""
  words = dict.fromkeys(word.casefold() for word in WORD.findall(query))
  kept = [word for word in words if len(word) >= MIN_TERM and word not in STOP_WORDS]
  return " OR ".join(f'"{word}"' for word in kept)


def search(db: Path, query: str, k: int = 3) -> list[Hit]:
  """The `k` best articles for `query` by BM25; none when no word is left to search."""
  expression = match_terms(query)
  if not expression or k < 1:
    return []
  with closing(_connect(db)) as connection:
    found = connection.execute(
      "SELECT origin_ref, name, number, text FROM search WHERE search MATCH ? "
      "ORDER BY rank LIMIT ?",
      (expression, k),
    ).fetchall()
  return [Hit(*row) for row in found]


def article(db: Path, origin_ref: str) -> Hit | None:
  """The indexed article cited as `origin_ref`, if any."""
  with closing(_connect(db)) as connection:
    row = connection.execute(
      "SELECT origin_ref, name, number, text FROM articles WHERE origin_ref = ? ORDER BY id",
      (origin_ref,),
    ).fetchone()
  return None if row is None else Hit(*row)
