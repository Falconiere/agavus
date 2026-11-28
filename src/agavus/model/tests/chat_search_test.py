"""Chat search over a synthetic corpus and index: domain statutes first, cited like Planalto."""

from pathlib import Path

from agavus.corpus.httpfixtures import sample_text
from agavus.corpus.statutes import STATUTES
from agavus.corpus.storage import CorpusStorage
from agavus.model.chat_search import ChatSearch, statute_names
from agavus.model.law_search import Hit, search
from agavus.model.planalto import PLANALTO_DIR, SITE
from agavus.model.tests.law_search_test import synthetic_index

LMP = STATUTES[0]
LMP_NAME = "Lei nº 11.340, de 7 de agosto de 2006"
PROTECTIVE = "Art. 22. O juiz poderá aplicar medidas protetivas fictícias ao agressor."


def _chat_search(tmp_path: Path, *, corpus: bool) -> ChatSearch:
  db = synthetic_index(tmp_path)
  capture = tmp_path / PLANALTO_DIR / "sources" / f"{LMP.url.removeprefix(SITE)}.html"
  capture.parent.mkdir(parents=True, exist_ok=True)
  capture.write_text(f"<p>{LMP_NAME.upper()}</p><p>Cria mecanismos.</p><p>Art. 1º Lei.</p>")
  storage = CorpusStorage(tmp_path) if corpus else None
  if storage is not None:
    storage.save_text(sample_text("lmp22", PROTECTIVE, f"{LMP.name}, art. 22"))
  return ChatSearch(tmp_path, db, storage, None)


def test_statute_names_come_from_captures_else_the_corpus_name(tmp_path: Path) -> None:
  _chat_search(tmp_path, corpus=False)
  names = statute_names(tmp_path)
  assert names[LMP.name] == (LMP.url, LMP_NAME)
  assert names[STATUTES[1].name] == (STATUTES[1].url, STATUTES[1].name)


def test_domain_statutes_come_first_cited_by_their_planalto_url(tmp_path: Path) -> None:
  chat = _chat_search(tmp_path, corpus=True)
  question = "Quem fica com a guarda e quais medidas protetivas o juiz aplica?"
  hits = chat.hits(question)
  assert hits[0] == Hit(f"{LMP.url}, art. 22", LMP_NAME, "22", PROTECTIVE)
  assert hits[1:] == search(tmp_path / "lab/text/planalto/search.db", question, 3)


def test_without_a_corpus_only_planalto_is_searched(tmp_path: Path) -> None:
  chat = _chat_search(tmp_path, corpus=False)
  question = "Quem fica com a guarda dos filhos?"
  assert chat.hits(question) == search(tmp_path / "lab/text/planalto/search.db", question, 3)
