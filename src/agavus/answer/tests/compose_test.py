"""Tests for answer composition (tests folder, synthetic corpus)."""

from pathlib import Path

from agavus.answer.compose import DISCLAIMER, compose
from agavus.corpus.httpfixtures import sample_text
from agavus.corpus.storage import CorpusStorage

CONFIDENT_M = {"F": 0.01, "M": 0.98, "O": 0.01}
AMBIGUOUS = {"F": 0.40, "M": 0.55, "O": 0.05}
CONFIDENT_O = {"F": 0.01, "M": 0.01, "O": 0.98}


def _corpus(tmp_path: Path) -> CorpusStorage:
  storage = CorpusStorage(tmp_path)
  storage.save_text(
    sample_text(
      doc_id="d1",
      origin_ref="Lei 11.340/2006, art. 22",
      text="A medida protetiva de urgência poderá ser concedida pelo juiz.",
      citations=("art. 22",),
    )
  )
  storage.rebuild_index()
  return storage


def test_accept_renders_cited_excerpt(tmp_path: Path) -> None:
  answer = compose("medida protetiva", CONFIDENT_M, _corpus(tmp_path))
  assert answer.decision == "accept-M"
  assert "Lei 11.340/2006, art. 22" in answer.text
  assert "medida protetiva" in answer.text
  assert DISCLAIMER in answer.text
  assert answer.sources


def test_defer_asks_clarification(tmp_path: Path) -> None:
  answer = compose("meu ex não deixa ver meu filho", AMBIGUOUS, _corpus(tmp_path))
  assert answer.decision == "defer"
  assert "esclarecimento" in answer.text
  assert answer.sources == ()


def test_reject_states_scope(tmp_path: Path) -> None:
  answer = compose("tráfico de drogas", CONFIDENT_O, _corpus(tmp_path))
  assert answer.decision == "reject"
  assert "Fora do escopo" in answer.text


def test_no_hits_degrades_honestly(tmp_path: Path) -> None:
  answer = compose("medida protetiva", CONFIDENT_M, CorpusStorage(tmp_path))
  assert answer.decision == "defer"
  assert "base segura" in answer.text
  assert answer.sources == ()


def test_irrelevant_hit_degrades(tmp_path: Path) -> None:
  answer = compose("alienação parental", CONFIDENT_M, _corpus(tmp_path))
  assert answer.decision == "defer"
  assert answer.sources == ()
