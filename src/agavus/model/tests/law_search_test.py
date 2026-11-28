"""BM25 search over synthetic Planalto-like articles: ranking, names, laws named by number."""

import json
from pathlib import Path

from agavus.corpus.records import TextSource
from agavus.corpus.statutes import STATUTES
from agavus.corpus.storage import CorpusStorage
from agavus.model.federal import PUBLIC_DOMAIN
from agavus.model.law_search import (
  CHAT_SEARCH_FILE,
  EMENTA,
  SEARCH_FILE,
  Hit,
  NamedLaw,
  article,
  build_chat_index,
  build_index,
  index_counts,
  match_terms,
  named_articles,
  named_law,
  search,
)
from agavus.model.planalto import PLANALTO_DIR, SITE
from agavus.model.tests.data_test import document

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
  # Only LAW has a card, so only LAW has an ementa row, cited by its url.
  assert article(db, LAW) == Hit(LAW, NAME, EMENTA, "Cria medidas fictícias.")
  assert article(db, OTHER) is None


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


def test_a_law_named_by_number_puts_its_article_or_ementa_first(tmp_path: Path) -> None:
  db = synthetic_index(tmp_path)
  # Without the law's number, BM25 ranks OTHER's divorce article first for these words.
  assert search(db, "O que diz o art. 3 sobre divórcio?")[0].origin_ref == f"{OTHER}, art. 1"
  asked = search(db, "O que diz o art. 3 da Lei 99999 sobre divórcio?", k=6)
  assert asked[0].origin_ref == f"{LAW}, art. 3"
  assert f"{OTHER}, art. 1" in [hit.origin_ref for hit in asked]
  assert search(db, "Do que trata a Lei nº 99.999?")[0] == article(db, LAW)
  covered = search(db, "A Lei 99.999 fala de guarda?", k=5)
  assert [hit.origin_ref for hit in covered[:2]] == [LAW, f"{LAW}, art. 2"]
  assert len({hit.origin_ref for hit in covered}) == len(covered)
  # An unknown law number falls back to plain BM25.
  assert search(db, "Lei 12.345 e a pensão alimentícia")[0].origin_ref == f"{LAW}, art. 3"


def test_law_and_article_numbers_are_read_the_way_names_write_them() -> None:
  assert named_law("art. 49 da Lei 8078/90") == NamedLaw("Lei nº 8.078,", "8.078")
  assert named_law("lei complementar n. 101") == NamedLaw("Lei Complementar nº 101,", "101")
  assert named_law("Decreto-Lei nº 6.887") == NamedLaw("Decreto-Lei nº 6.887,", "6.887")
  assert named_law("a lei de 1990") is None and named_law("alienação parental") is None
  assert named_law("art. 22 da Lei Maria da Penha") == NamedLaw("Lei nº 11.340,", "11.340")
  assert named_law("o que diz o codigo civil?") == NamedLaw("Lei nº 10.406,", "10.406")
  assert named_articles("artigo 1.024 do Código") == ["1.024", "1.024º"]
  assert named_articles("Art. 12-c") == ["12-C", "12º-C"]
  assert named_articles("Do que trata a lei?") == []
  assert match_terms("Art. 2 da Lei 8.078") == '"8.078"'


LMP = STATUTES[0]
LMP_NAME = "Lei nº 11.340, de 7 de agosto de 2006"
PROTECTIVE = "Art. 22. O juiz poderá aplicar medidas protetivas fictícias de urgência ao agressor."


def test_the_chat_index_adds_exam_statutes_that_training_never_sees(tmp_path: Path) -> None:
  synthetic_index(tmp_path)
  folder = tmp_path / PLANALTO_DIR
  page = f"<p>{LMP_NAME.upper()}</p><p>Cria mecanismos fictícios.</p><p>Art. 1º Esta Lei.</p>"
  (folder / "sources" / f"{LMP.url.removeprefix(SITE)}.html").parent.mkdir(parents=True)
  (folder / "sources" / f"{LMP.url.removeprefix(SITE)}.html").write_text(page, encoding="utf-8")
  crawled = {"license": PUBLIC_DOMAIN, "url": LMP.url, "origin_ref": f"{LMP.url}, art. 46"}
  crawl = folder / "articles" / "lmp.jsonl"
  crawl.write_text(json.dumps(crawled | {"text": "Art. 46. Vigência fictícia."}) + "\n")
  CorpusStorage(tmp_path).save_texts(
    [document(f"{LMP.name}, art. 22", TextSource.STATUTE, PROTECTIVE)]
  )
  training, chat = build_index(tmp_path), build_chat_index(tmp_path)
  assert chat == tmp_path / CHAT_SEARCH_FILE
  question = "O que diz o art. 22 da Lei 11.340 sobre medidas protetivas?"
  assert search(chat, question)[0] == Hit(f"{LMP.url}, art. 22", LMP_NAME, "22", PROTECTIVE)
  assert article(chat, LMP.url) == Hit(LMP.url, LMP_NAME, EMENTA, "Cria mecanismos fictícios.")
  assert search(chat, "art. 22 da Lei Maria da Penha")[0].number == "22"
  # Training keeps only the crawl; the chat adds the corpus articles and keeps the rest.
  assert all("fictícias de urgência" not in hit.text for hit in search(training, question, k=9))
  assert article(training, f"{LMP.url}, art. 46") == article(chat, f"{LMP.url}, art. 46")
  assert index_counts(training) == (6, 3) and index_counts(chat) == (7, 3)
