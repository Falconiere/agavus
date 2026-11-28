"""BM25 search over every Planalto article, so Agathe answers from retrieved law (§9).

The index is SQLite FTS5 with accents folded. Rows keep the verbatim `origin_ref` and the
law's citable name; answers and chat sources cite them as they are. Each law's ementa is a
row too, cited by the law's url with an empty article number. A query that names a law by
number ("Lei 8.078", "art. 49 da Lei nº 8.078") gets that law's article, or its ementa, first.
"""

import re
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from agavus.corpus.statutes import LAW_ALIASES, STATUTES, TOPICLESS_NAMES
from agavus.model.data import load_statute_articles
from agavus.model.planalto import PLANALTO_DIR, SITE
from agavus.model.sft_rules import LawCard, article_number, law_card, load_rule_sources

SEARCH_FILE = PLANALTO_DIR / "search.db"
# The chat's index adds the exam statutes; training never reads it.
CHAT_SEARCH_FILE = PLANALTO_DIR / "search-chat.db"
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


Row = tuple[str, str, str, str, str]


def _planalto_rows(data_dir: Path) -> list[Row]:
  """(origin_ref, url, name, number, text) for every crawled article, then every ementa."""
  cards, articles = load_rule_sources(data_dir)
  names = {card.url: card.name for card in cards}
  rows = [
    (origin_ref, url, names.get(url, url), article_number(origin_ref, text), text)
    for url, found in sorted(articles.items())
    for origin_ref, text in found
  ]
  ementas = [(card.url, card.url, card.name, EMENTA, card.ementa) for card in cards if card.ementa]
  return rows + ementas


def _statute_rows(data_dir: Path) -> list[Row]:
  """The exam statutes whole, from the corpus, cited by their Planalto url like other rows.

  Each statute's name and ementa come from its Planalto capture.
  """
  folder, urls = data_dir / PLANALTO_DIR, {ref.name: ref.url for ref in STATUTES}
  cards: dict[str, LawCard] = {}
  for url in urls.values():
    capture = folder / "sources" / f"{url.removeprefix(SITE)}.html"
    card = law_card(url, capture.read_bytes()) if capture.is_file() else None
    if card is not None:
      cards[url] = card
  rows: list[Row] = []
  for origin_ref, text in load_statute_articles(data_dir):
    law, _, number = origin_ref.partition(", art. ")
    page = urls.get(law)
    if page is not None:
      name = cards[page].name if page in cards else law
      rows.append((f"{page}, art. {number}", page, name, article_number(origin_ref, text), text))
  return rows + [(url, url, card.name, EMENTA, card.ementa) for url, card in cards.items()]


def _write_index(path: Path, rows: list[Row]) -> Path:
  """A fresh index of `rows` at `path`, replacing the old one at once."""
  temporary = path.with_name(path.name + ".tmp")
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary.unlink(missing_ok=True)
  with closing(sqlite3.connect(temporary)) as connection, connection:
    connection.executescript(SCHEMA)
    connection.executemany(
      "INSERT INTO articles (origin_ref, url, name, number, text) VALUES (?, ?, ?, ?, ?)", rows
    )
    connection.execute("INSERT INTO search (search) VALUES ('rebuild')")
  temporary.replace(path)
  return path


def build_index(data_dir: Path) -> Path:
  """The training index: every crawled Planalto article and ementa, exam statutes left out.

  `build-sft-context` reads only this one, so no exam text reaches training.
  """
  return _write_index(data_dir / SEARCH_FILE, _planalto_rows(data_dir))


def build_chat_index(data_dir: Path) -> Path:
  """The chat index: the training rows plus the exam statutes' articles and ementas.

  Retrieval only shows articles to the reader; it trains nothing, so the exam stays clean.
  """
  statutes = _statute_rows(data_dir)
  # The corpus holds only part of a code (the Civil Code's família book): replace article by
  # article, so the rest of the crawl stays.
  replaced = {(row[1], row[3]) for row in statutes}
  kept = [row for row in _planalto_rows(data_dir) if (row[1], row[3]) not in replaced]
  return _write_index(data_dir / CHAT_SEARCH_FILE, kept + statutes)


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


def _folded(text: str) -> str:
  """Casefolded without accents, so 'Codigo Civil' matches 'código civil'."""
  decomposed = unicodedata.normalize("NFKD", text.casefold())
  return "".join(char for char in decomposed if not unicodedata.combining(char))


def _popular(alias: str) -> NamedLaw:
  """The NamedLaw for a popular name, from its alias ('Lei 11.340/2006')."""
  number = LAW_ALIASES[alias].removeprefix("Lei ").partition("/")[0]
  return NamedLaw(f"Lei nº {number},", number)


def named_law(query: str) -> NamedLaw | None:
  """The law the query names by kind and number ('Lei 8078', 'Decreto-Lei nº 6.887') or by a
  popular name that says which law ('Lei Maria da Penha', 'Código Civil'), if any."""
  found = LAW_NAMED.search(query)
  if found is None:
    folded = _folded(query)
    named = [alias for alias in TOPICLESS_NAMES if _folded(alias).removeprefix("lei ") in folded]
    return _popular(named[0]) if named else None
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


def labeled(db: Path, name: str, number: str) -> Hit | None:
  """The indexed row a context label names: law name and article number (EMENTA for ementa)."""
  with closing(_connect(db)) as connection:
    row = connection.execute(
      "SELECT origin_ref, name, number, text FROM articles WHERE name = ? AND number = ? "
      "ORDER BY id LIMIT 1",
      (name, number),
    ).fetchone()
  return None if row is None else Hit(*row)
