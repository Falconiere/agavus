"""Extractive answers with a keyword reranker: passages, the best one, the quoted answer."""

import re
from collections.abc import Sequence

from agavus.model.law_search import EMENTA, Hit
from agavus.model.passages import (
  ANSWER_CHARS,
  MIN_SCORE,
  Choice,
  Extractor,
  Passage,
  asked,
  best_passage,
  passages,
  quoted,
)
from agavus.model.sft_context import NOT_FOUND

NAME = "Lei nº 99.999, de 1º de janeiro de 2030"
GUARD = Hit(
  "l99999, art. 2",
  NAME,
  "2º",
  "Art. 2º A guarda segue o acordo dos pais. § 1º Sem acordo, decide o juiz. "
  "Parágrafo único. Ouve-se o Ministério Público.",
)
# Words this short ("de", "o") match every citation, so the fake ignores them.
SHORT = 2
EMENTA_HIT = Hit("l99999", NAME, EMENTA, "Cria medidas fictícias de proteção.")


class KeywordReranker:
  """Scores a passage by how many question words it holds; remembers what it saw."""

  def __init__(self) -> None:
    self.seen: list[str] = []

  def scores(self, question: str, texts: Sequence[str]) -> list[float]:
    """Shared words longer than two letters; on a tie, the shorter passage wins."""
    self.seen.extend(texts)
    words = {word for word in re.findall(r"\w+", question.casefold()) if len(word) > SHORT}
    found = [set(re.findall(r"\w+", text.casefold())) for text in texts]
    return [len(words & seen) - len(text) / 1e4 for seen, text in zip(found, texts, strict=True)]


class FixedFinder:
  """Always finds the same articles."""

  def hits(self, question: str) -> list[Hit]:
    """GUARD and the ementa, whatever the question."""
    return [GUARD, EMENTA_HIT]


def test_an_article_splits_into_whole_caput_and_paragraphs() -> None:
  texts = [passage.text for passage in passages(GUARD)]
  assert texts == [
    GUARD.text,
    "Art. 2º A guarda segue o acordo dos pais.",
    "§ 1º Sem acordo, decide o juiz.",
    "Parágrafo único. Ouve-se o Ministério Público.",
  ]
  assert [passage.text for passage in passages(EMENTA_HIT)] == [EMENTA_HIT.text]


def test_the_best_passage_is_quoted_verbatim_with_its_source() -> None:
  reranker = KeywordReranker()
  choice = best_passage("sem acordo quem decide? o juiz", [GUARD, EMENTA_HIT], reranker)
  assert choice is not None and choice.passage.text == "§ 1º Sem acordo, decide o juiz."
  # The reranker sees each passage under its citation.
  assert f"{NAME}, ementa: {EMENTA_HIT.text}" in reranker.seen
  assert quoted(choice) == f"§ 1º Sem acordo, decide o juiz.\n\nFonte: {NAME}, art. 2º."
  assert best_passage("x", [], reranker) is None


def test_low_scores_defer_and_long_quotes_are_cut() -> None:
  assert quoted(None) == NOT_FOUND
  assert quoted(Choice(Passage(GUARD, GUARD.text), MIN_SCORE - 0.01)) == NOT_FOUND
  assert quoted(Choice(Passage(EMENTA_HIT, EMENTA_HIT.text), MIN_SCORE)).endswith(f"Fonte: {NAME}.")
  long = Passage(GUARD, "palavra " * ANSWER_CHARS)
  text, source = quoted(Choice(long, MIN_SCORE)).split("\n\n")
  assert len(text) <= ANSWER_CHARS + 2 and text.endswith("palavra …")
  assert source == f"Fonte: {NAME}, art. 2º."


def test_the_extractor_answers_from_what_it_finds() -> None:
  extractor = Extractor(FixedFinder(), KeywordReranker())
  answer, hits, choice = extractor.answer("cria medidas fictícias de proteção")
  assert hits == [GUARD, EMENTA_HIT] and choice is not None
  assert answer == f"{EMENTA_HIT.text}\n\nFonte: {NAME}."
  assert extractor.answer("bolo de cenoura")[0] == NOT_FOUND


def test_a_question_naming_law_and_article_quotes_only_that_article() -> None:
  mention = Hit(
    "l11111, art. 1", "Lei nº 11.111, de 1º de março de 2031", "1º", "Altera o art. 2º."
  )
  assert asked("O que diz o art. 2 da Lei 99.999?", [mention, GUARD, EMENTA_HIT]) == [GUARD]
  assert asked("O que diz a Lei 99.999?", [GUARD]) == []
  assert asked("guarda", [GUARD]) == []
  spaced = Hit("r", NAME, "3º", "Art. 3º  Um\t\ttexto\n  espaçado.")
  assert [passage.text for passage in passages(spaced)] == ["Art. 3º Um texto espaçado."]


class DoubtfulReranker:
  """Scores every passage below MIN_SCORE."""

  def scores(self, question: str, texts: Sequence[str]) -> list[float]:
    """The same low score for all."""
    return [MIN_SCORE - 5] * len(texts)


def test_a_named_article_is_quoted_whole_even_with_a_low_score() -> None:
  extractor = Extractor(FixedFinder(), DoubtfulReranker())
  assert extractor.answer("guarda?")[0] == NOT_FOUND
  answer, _, choice = extractor.answer("art. 2 da Lei 99.999?")
  assert choice is not None and choice.lookup and choice.score < MIN_SCORE
  assert answer == f"{GUARD.text}\n\nFonte: {NAME}, art. 2º."
