"""Grounding of chat answers in the shown articles: coverage, citations, deferral."""

from agavus.model.grounding import (
  GROUNDED_SHARE,
  checked,
  cites_asked,
  cites_shown,
  coverage,
  grounded,
)
from agavus.model.law_search import EMENTA, Hit
from agavus.model.sft_context import NOT_FOUND

NAME = "Lei nº 99.999, de 1º de janeiro de 2030"
GUARD = "Art. 2º A guarda dos filhos segue o acordo dos pais, ouvido o Ministério Público."
HITS = [
  Hit("l99999, art. 2", NAME, "2º", GUARD),
  Hit("l99999", NAME, EMENTA, "Cria medidas fictícias de proteção."),
  Hit("l88888, art. 1.024", "Lei nº 88.888, de 2 de março de 2031", "1.024", "Art. 1.024 O bem."),
]


def test_a_quoted_article_with_its_citation_is_grounded() -> None:
  answer = f"{GUARD}\n\nFonte: {NAME}, art. 2º."
  assert coverage(answer, HITS) == 1.0 and grounded(answer, HITS)
  assert checked(answer, HITS) == answer
  # A quote cut short still copies the article.
  assert grounded("Art. 2º A guarda dos filhos segue o acordo", HITS)


def test_ementa_reverse_and_deferral_answers_are_grounded() -> None:
  assert grounded(f"Cria medidas fictícias de proteção.\n\nFonte: {NAME}.", HITS)
  assert grounded(f"{NAME}.", HITS)
  assert grounded(NOT_FOUND, HITS) and grounded(NOT_FOUND, [])
  assert grounded(
    "Art. 1.024 O bem.\n\nFonte: Lei nº 88.888, de 2 de março de 2031, art. 1.024.", HITS
  )


def test_invented_text_defers() -> None:
  invented = f"A revogação da medida ocorre a partir de 14 de abril.\n\nFonte: {NAME}, art. 2º."
  assert coverage(invented, HITS) < GROUNDED_SHARE
  assert checked(invented, HITS) == NOT_FOUND
  assert checked("", HITS) == NOT_FOUND and coverage("", HITS) == 0.0
  assert checked(f"{GUARD}\n\nFonte: {NAME}, art. 2º.", []) == NOT_FOUND


def test_the_citation_must_name_a_shown_law_and_article() -> None:
  assert cites_shown("Sem fonte.", HITS)
  assert cites_shown(f"x\n\nFonte: {NAME}, art. 2º.", HITS)
  assert not cites_shown(f"x\n\nFonte: {NAME}, art. 7.", HITS)
  assert not cites_shown("x\n\nFonte: Lei nº 11.340, de 7 de agosto de 2006, art. 22.", HITS)
  assert not grounded(f"{GUARD}\n\nFonte: Lei nº 1, de 1º de maio de 1900, art. 2º.", HITS)


def test_a_question_naming_an_article_needs_that_article_cited() -> None:
  other = f"Cria medidas fictícias de proteção.\n\nFonte: {NAME}."
  quoted = f"{GUARD}\n\nFonte: {NAME}, art. 2º."
  assert grounded(quoted, HITS, "O que diz o art. 2 da Lei 99.999?")
  # A real excerpt of the wrong article, or of the law asked without its article, defers.
  assert not grounded(quoted, HITS, "O que diz o art. 3 da Lei 99.999?")
  assert not grounded(other, HITS, "O que diz o art. 2 da Lei 99.999?")
  assert not grounded(quoted, HITS, "O que diz o art. 2 da Lei 11.340?")
  assert grounded(other, HITS, "Do que trata a Lei 99999?")
  assert not cites_asked(f"{NAME}.", "Do que trata a Lei 99.999?")
  assert cites_asked(quoted, "Quem fica com a guarda?") and grounded(NOT_FOUND, HITS, "art. 9")
