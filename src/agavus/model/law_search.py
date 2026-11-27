"""BM25 search over every Planalto article, so Agathe answers from retrieved law (§9).

The index is SQLite FTS5 with accents folded. Rows keep the verbatim `origin_ref` and the
law's citable name; answers and chat sources cite them as they are. Each law's ementa is a
row too, cited by the law's url with an empty article number. A query that names a law by
number ("Lei 8.078", "art. 49 da Lei nº 8.078") gets that law's article, or its ementa, first.
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
  # In every law's name or article: no ranking signal, and each one scans the whole index.
  | {"lei", "art", "arts", "artigo", "nº", "nºs"}
)
# Shorter words are dropped from queries: single letters match nearly every article.
MIN_TERM = 2
# Dotted numbers stay whole, so "8.078" searches the phrase "8 078" and not every "078".
WORD = re.compile(r"\d+(?:\.\d+)+|\w+")
LAW_NAMED = re.compile(
  r"(?i)\b(lei\s+complementar|lei\s+delegada|decreto-lei|lei)\s+(?:n\s*[º°o.]*\s*)?"
  r"(\d{1,3}(?:\.\d{3})+|\d+)(?!\d)"
)
ARTICLE_NAMED = re.compile(r"(?i)\bart(?:igo|\.)?\s*(\d{1,3}(?:\.\d{3})+|\d+)\s*[º°o]?(-[a-z]+)?")
KIND_NAMES = {"lei complementar": "Lei Complementar", "lei delegada": "Lei Delegada"}
KIND_NAMES |= {"decreto-lei": "Decreto-Lei", "lei": "Lei"}
# Rows fetched per wanted hit within a named law, before other kinds sharing its number drop.
LAW_SLACK = 4
# An ementa row has no article number.
EMENTA = ""
SCHEMA = """
CREATE TABLE articles (
  id INTEGER PRIMARY KEY, origin_ref TEXT, url TEXT, name TEXT, number TEXT, text TEXT
);
CREATE INDEX articles_by_ref ON articles (origin_ref);
CREATE INDEX articles_by_name ON articles (name, number);
CREATE VIRTUAL TABLE search USING fts5 (
  origin_ref UNINDEXED, url UNINDEXED, name, number, text,
  content = 'articles', content_rowid = 'id', tokenize = 'unicode61 remove_diacritics 2'
);
"""


@dataclass(frozen=True)
class Hit:
  """One article found: its citation, the law's name, the article number and its text.

  An ementa hit has the number EMENTA and the law's url as its citation.
  """

  origin_ref: str
  name: str
  number: str
  text: str


def build_index(data_dir: Path) -> Path:
  """Index every crawled Planalto article and ementa; rebuilt whole, replaced at once."""
  cards, articles = load_rule_sources(data_dir)
  names = {card.url: card.name for card in cards}
  ementas = [(card.url, card.url, card.name, EMENTA, card.ementa) for card in cards if card.ementa]
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
    connection.executemany(
      "INSERT INTO articles (origin_ref, url, name, number, text) VALUES (?, ?, ?, ?, ?)", ementas
    )
    connection.execute("INSERT INTO search (search) VALUES ('rebuild')")
  temporary.replace(path)
  return path


def _connect(db: Path) -> sqlite3.Connection:
  """Read-only, so a missing index raises instead of becoming an empty file."""
  return sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)


def index_counts(db: Path) -> tuple[int, int]:
  """(articles, laws) in the index; ementa rows are not articles."""
  with closing(_connect(db)) as connection:
    articles, laws = connection.execute(
      "SELECT coalesce(sum(number != ?), 0), count(DISTINCT url) FROM articles", (EMENTA,)
    ).fetchone()
  return int(articles), int(laws)


def match_terms(query: str) -> str:
  """The query as an OR of quoted words, stop words and single letters dropped."""
  words = dict.fromkeys(word.casefold() for word in WORD.findall(query))
  kept = [word for word in words if len(word) >= MIN_TERM and word not in STOP_WORDS]
  return " OR ".join(f'"{word}"' for word in kept)


def _thousands(number: str) -> str:
  """'8078' and '8.078' both as '8.078', the way law names and articles write them."""
  return f"{int(number.replace('.', '')):,}".replace(",", ".")


@dataclass(frozen=True)
class NamedLaw:
  """A law the query names: how its name starts ('Lei nº 8.078,') and its number."""

  prefix: str
  number: str


def named_law(query: str) -> NamedLaw | None:
  """The law the query names by kind and number ('Lei 8078', 'Decreto-Lei nº 6.887'), if any."""
  found = LAW_NAMED.search(query)
  if found is None:
    return None
  kind = KIND_NAMES[" ".join(found.group(1).casefold().split())]
  number = _thousands(found.group(2))
  return NamedLaw(f"{kind} nº {number},", number)


def named_articles(query: str) -> list[str]:
  """Numbers the query's 'art. N' may be written as in the index ('1º', '49', '12-C')."""
  found = ARTICLE_NAMED.search(query)
  if found is None:
    return []
  number, suffix = _thousands(found.group(1)), (found.group(2) or "").upper()
  return [number + suffix, number + "º" + suffix]


def _pinned(connection: sqlite3.Connection, law: NamedLaw, numbers: list[str]) -> list[Hit]:
  """The named law's asked article, or its ementa when no article is asked."""
  wanted = numbers or [EMENTA]
  marks = ", ".join("?" * len(wanted))
  rows = connection.execute(
    "SELECT origin_ref, name, number, text FROM articles "
    f"WHERE name GLOB ? AND number IN ({marks}) ORDER BY id LIMIT 1",
    (law.prefix + "*", *wanted),
  ).fetchall()
  return [Hit(*row) for row in rows]


def _ranked(
  connection: sqlite3.Connection, expression: str, law: NamedLaw | None, k: int
) -> list[Hit]:
  """BM25 hits, only within `law` when given.

  The law's number as a phrase in the name column keeps the match an index intersection;
  the name prefix then drops other kinds with the same number.
  """
  if law is not None:
    expression = f'({expression}) AND name : "{law.number}"'
  rows = connection.execute(
    "SELECT origin_ref, name, number, text FROM search WHERE search MATCH ? ORDER BY rank LIMIT ?",
    (expression, k if law is None else k * LAW_SLACK),
  ).fetchall()
  hits = [Hit(*row) for row in rows]
  return hits if law is None else [hit for hit in hits if hit.name.startswith(law.prefix)][:k]


def search(db: Path, query: str, k: int = 3) -> list[Hit]:
  """The `k` best articles for `query`; a law named by number comes first, then BM25."""
  expression, law = match_terms(query), named_law(query)
  if k < 1 or not (expression or law):
    return []
  with closing(_connect(db)) as connection:
    found = [] if law is None else _pinned(connection, law, named_articles(query))
    if expression and law is not None:
      found += _ranked(connection, expression, law, k)
    if expression:
      found += _ranked(connection, expression, None, k)
  unique: dict[str, Hit] = {}
  for hit in found:
    unique.setdefault(hit.origin_ref, hit)
  return list(unique.values())[:k]


def article(db: Path, origin_ref: str) -> Hit | None:
  """The indexed article cited as `origin_ref`, if any."""
  with closing(_connect(db)) as connection:
    row = connection.execute(
      "SELECT origin_ref, name, number, text FROM articles WHERE origin_ref = ? ORDER BY id",
      (origin_ref,),
    ).fetchone()
  return None if row is None else Hit(*row)
