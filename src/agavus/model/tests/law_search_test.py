"""BM25 search over synthetic Planalto-like articles: ranking, names, unsafe queries."""

import json
from pathlib import Path

from agavus.model.federal import PUBLIC_DOMAIN
from agavus.model.law_search import (
  SEARCH_FILE,
  Hit,
  article,
  build_index,
  index_counts,
  match_terms,
  search,
)
from agavus.model.planalto import PLANALTO_DIR, SITE

LAW = SITE + "leis/l99999.htm"
OTHER = SITE + "leis/l88888.htm"
NAME = "Lei nº 99.999, de 1º de janeiro de 2030"
PAGE = f"<p>{NAME.upper()}</p><p>Cria medidas fictícias.</p><p>Art. 1º Esta Lei.</p>"
ARTICLES = {
  LAW: [
    "Art. 1º Esta Lei cria medidas protetivas fictícias para a vítima.",
    "Art. 2º A guarda dos filhos segue o acordo dos pais.",
    "Art. 3º O juiz decide a pensão alimentícia em cinco dias.",
  ],
  OTHER: [
    "Art. 1º A ação de divórcio tramita na vara de família.",
    "Art. 2º O regime de bens se escolhe antes do casamento.",
  ],
}


def synthetic_index(data_dir: Path) -> Path:
  """A search index over ARTICLES; only LAW has a captured page, so OTHER keeps its url."""
  folder = data_dir / PLANALTO_DIR
  capture = folder / "sources" / f"{LAW.removeprefix(SITE)}.html"
  capture.parent.mkdir(parents=True)
  capture.write_text(PAGE, encoding="utf-8")
  for url, texts in ARTICLES.items():
    path = folder / "articles" / f"{url.removeprefix(SITE)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
      {"license": PUBLIC_DOMAIN, "url": url, "origin_ref": f"{url}, art. {n}", "text": text}
      for n, text in enumerate(texts, 1)
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
  return build_index(data_dir)


def test_index_names_laws_and_rebuilds_in_place(tmp_path: Path) -> None:
  db = synthetic_index(tmp_path)
  assert db == tmp_path / SEARCH_FILE
  assert index_counts(db) == (5, 2)
  assert build_index(tmp_path) == db and index_counts(db) == (5, 2)
  assert not list(db.parent.glob("*.tmp"))
  assert article(db, f"{LAW}, art. 2") == Hit(f"{LAW}, art. 2", NAME, "2º", ARTICLES[LAW][1])
  assert article(db, f"{OTHER}, art. 1") == Hit(f"{OTHER}, art. 1", OTHER, "1º", ARTICLES[OTHER][0])
  assert article(db, f"{LAW}, art. 9") is None


def test_search_ranks_the_matching_article_first_without_accents(tmp_path: Path) -> None:
  db = synthetic_index(tmp_path)
  assert search(db, "Como funciona a pensao alimenticia?")[0].origin_ref == f"{LAW}, art. 3"
  assert search(db, "DIVÓRCIO!")[0].origin_ref == f"{OTHER}, art. 1"
  assert len(search(db, "lei medidas guarda divórcio bens", k=1)) == 1


def test_search_survives_fts_syntax_and_empty_queries(tmp_path: Path) -> None:
  db = synthetic_index(tmp_path)
  assert match_terms('Guarda "dos" filhos? (NEAR) a-b * e') == '"guarda" OR "filhos" OR "near"'
  for query in ['"', "guarda AND OR NOT (", "art.: ^*", 'x" OR "y', "NEAR(guarda filhos)"]:
    search(db, query)
  assert search(db, "") == search(db, "de a o e?!") == search(db, "guarda", k=0) == []
