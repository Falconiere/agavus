"""Tests for answer composition (tests folder, synthetic corpus)."""

from dataclasses import replace
from pathlib import Path

from agavus.answer.compose import DISCLAIMER, compose, verify_deliverable
from agavus.corpus.httpfixtures import sample_text
from agavus.corpus.records import TextSource
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


def test_verified_chain_answer(tmp_path: Path) -> None:
  storage = _corpus(tmp_path)
  storage.save_text(
    sample_text(
      doc_id="p24a",
      origin_ref="Lei 11.340/2006, art. 24-A",
      text="Descumprir decisão judicial que defere medidas protetivas de urgência.",
    )
  )
  answer = compose("ele descumpriu a medida protetiva", CONFIDENT_M, storage)
  assert answer.decision == "accept-M"
  assert "cadeia verificada" in answer.text
  assert "R-24A" in answer.text
  assert answer.sources == ("Lei 11.340/2006, art. 24-A",)
  assert DISCLAIMER in answer.text


def test_broken_chain_defers(tmp_path: Path) -> None:
  answer = compose("ele descumpriu a medida protetiva", CONFIDENT_M, _corpus(tmp_path))
  assert answer.decision == "defer"
  assert answer.sources == ()


def test_domain_clear_defer_asks_topic(tmp_path: Path) -> None:
  probs = {"F": 0.20, "M": 0.70, "O": 0.10}
  answer = compose("qual é o artigo da lei da maria da penha", probs, _corpus(tmp_path))
  assert answer.decision == "defer"
  assert "tema específico" in answer.text
  assert "Maria da Penha" in answer.text


def test_stopword_only_overlap_rejected_as_irrelevant(tmp_path: Path) -> None:
  answer = compose("alienação de bens", CONFIDENT_M, _corpus(tmp_path))
  assert answer.decision == "defer"
  assert answer.sources == ()


def test_verify_deliverable_checks_advice_and_disclaimer() -> None:
  assert verify_deliverable(f"Trecho legal.\n{DISCLAIMER}", ("Lei X, art. 1",))
  assert not verify_deliverable(f"Você deve processar.\n{DISCLAIMER}", ("Lei X, art. 1",))
  assert not verify_deliverable("Trecho sem aviso.", ("Lei X, art. 1",))
  assert not verify_deliverable(f"Trecho.\n{DISCLAIMER}", ())


def _add_decision(storage: CorpusStorage, text: str) -> None:
  decision = sample_text(doc_id="dec", origin_ref="Decisão sintética", text=text)
  storage.save_text(replace(decision, source=TextSource.DJEN))


def test_statutes_answer_before_decisions(tmp_path: Path) -> None:
  storage = _corpus(tmp_path)
  _add_decision(storage, "medida protetiva medida protetiva medida protetiva")
  answer = compose("medida protetiva", CONFIDENT_M, storage)
  assert answer.sources == ("Lei 11.340/2006, art. 22",)


def test_decisions_answer_when_no_statute_matches(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  _add_decision(storage, "medida protetiva")
  answer = compose("medida protetiva", CONFIDENT_M, storage)
  assert answer.sources == ("Decisão sintética",)
