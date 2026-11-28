"""Extractive chat answers: the best passage of the articles found, quoted verbatim (§9).

Answers are cited excerpts, never generated text. Each article found is split into passages
(the whole article, its caput with incisos, each paragraph); a pretrained reranker scores
them against the question, and the best one is shown word for word with a citation built
from the article it came from. Below MIN_SCORE the chat defers instead.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from agavus.model.law_search import EMENTA, Hit, named_articles, named_law
from agavus.model.reranker import Reranker
from agavus.model.sft_context import NOT_FOUND, cited

# Paragraph boundaries inside an article: "§ 1º", "§ 10", "Parágrafo único".
PARAGRAPH = re.compile(r"\s(?=§\s*\d|Parágrafo único)")
# Longest quote shown; longer passages end with "…".
ANSWER_CHARS = 2000
# Reranker logit below which the chat defers. On 600 natural teacher questions (2026-11-28)
# it kept 96% of the answerable ones; off-topic questions score far below (about -11).
MIN_SCORE = 1.0


@dataclass(frozen=True)
class Passage:
  """A verbatim piece of one article found."""

  hit: Hit
  text: str


@dataclass(frozen=True)
class Choice:
  """The best passage and its reranker score; `lookup` when the question named the article."""

  passage: Passage
  score: float
  lookup: bool = False


def passages(hit: Hit) -> list[Passage]:
  """The whole article, then each part split at its paragraphs, without repeats.

  Runs of spaces, tabs and line breaks become one space; the words stay verbatim.
  """
  parts = [hit.text, *PARAGRAPH.split(hit.text)]
  texts = dict.fromkeys(" ".join(part.split()) for part in parts if part.strip())
  return [Passage(hit, text) for text in texts]


def asked(question: str, hits: Sequence[Hit]) -> list[Hit]:
  """The hits a question names by law and article ("art. 22 da Lei Maria da Penha"), if found.

  Such a question is a lookup: the article is quoted whole whatever its score, and an article
  that merely mentions the asked one never wins.
  """
  law, numbers = named_law(question), named_articles(question)
  if law is None or not numbers:
    return []
  return [hit for hit in hits if hit.name.startswith(law.prefix) and hit.number in numbers]


def best_passage(question: str, hits: Sequence[Hit], reranker: Reranker) -> Choice | None:
  """The passage the reranker scores highest; None when there is nothing to score.

  The reranker sees each passage under its citation, so an ementa knows which law it is.
  """
  found = [passage for hit in hits for passage in passages(hit)]
  if not found:
    return None
  scores = reranker.scores(question, [f"{cited(p.hit)}: {p.text}" for p in found])
  best = max(range(len(found)), key=scores.__getitem__)
  return Choice(found[best], scores[best])


def _clipped(text: str) -> str:
  """At most ANSWER_CHARS, cut at a word and marked when cut."""
  if len(text) <= ANSWER_CHARS:
    return text
  return text[:ANSWER_CHARS].rsplit(" ", 1)[0] + " …"


def quoted(choice: Choice | None) -> str:
  """The chat answer: the passage and its source, or NOT_FOUND below MIN_SCORE."""
  if choice is None or (choice.score < MIN_SCORE and not choice.lookup):
    return NOT_FOUND
  hit = choice.passage.hit
  source = hit.name if hit.number == EMENTA else f"{hit.name}, art. {hit.number}"
  return f"{_clipped(choice.passage.text)}\n\nFonte: {source}."


class Finder(Protocol):
  """Anything that finds the articles a question may be answered from."""

  def hits(self, question: str) -> list[Hit]:
    """Articles to quote from, best first."""
    ...


@dataclass(frozen=True)
class Extractor:
  """Search and reranker together: a question in, a quoted answer and its sources out."""

  finder: Finder
  reranker: Reranker

  def answer(self, question: str) -> tuple[str, list[Hit], Choice | None]:
    """The quoted answer (or NOT_FOUND), the articles consulted and the chosen passage."""
    hits = self.finder.hits(question)
    named = asked(question, hits)
    if named:
      whole = passages(named[0])[0]
      score = self.reranker.scores(question, [f"{cited(whole.hit)}: {whole.text}"])[0]
      choice: Choice | None = Choice(whole, score, lookup=True)
    else:
      choice = best_passage(question, hits, self.reranker)
    return quoted(choice), hits, choice
