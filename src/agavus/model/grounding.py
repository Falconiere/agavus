"""Whether a chat answer is grounded in the articles placed in its context (docs/llm-lab.md §9).

Answers are cited excerpts, never generated opinions. A context-tuned answer passes only when
most of its word 4-grams appear in the shown articles (full text and names), its "Fonte:"
line, if any, cites one of them, and, when the question names a law or article by number,
that is the one cited; anything else becomes the NOT_FOUND deferral.
On the 2026-11-27 context data, rule answers cover 1.0 of their articles; the two invented
chat answers covered 0.09 and 0.20.
"""

import re
from collections.abc import Sequence

from agavus.model.law_search import Hit, named_articles, named_law
from agavus.model.sft_context import NOT_FOUND

GRAM = 4
# Share of the answer's word 4-grams that must appear in the shown articles.
GROUNDED_SHARE = 0.8
SOURCE = "\n\nFonte:"
CITATION = re.compile(
  r"^\s*(?P<name>.+?)(?:,\s*art\.\s*(?P<number>\d+(?:\.\d{3})*[º°]?(?:-[A-Za-z]+)?))?\.?\s*$"
)
WORD = re.compile(r"\w+")


def _words(text: str) -> list[str]:
  return [word.casefold() for word in WORD.findall(text)]


def _grams(words: Sequence[str], size: int) -> list[tuple[str, ...]]:
  return [tuple(words[start : start + size]) for start in range(len(words) - size + 1)]


def coverage(answer: str, hits: Sequence[Hit]) -> float:
  """Share of the answer body's word n-grams found in the hits' names and texts.

  The body is the answer before its "Fonte:" line; answers shorter than GRAM words use
  their own length. An empty body covers nothing.
  """
  words = _words(answer.partition(SOURCE)[0])
  if not words:
    return 0.0
  size = min(GRAM, len(words))
  shown = {gram for hit in hits for gram in _grams(_words(f"{hit.name} {hit.text}"), size)}
  found = _grams(words, size)
  return sum(gram in shown for gram in found) / len(found)


def _citation(answer: str) -> tuple[str, str | None] | None:
  """(law name, article number or None) from the "Fonte:" line, if it has one."""
  _, marker, cited = answer.partition(SOURCE)
  found = CITATION.match(cited) if marker else None
  return None if found is None else (found["name"].strip(), found["number"])


def cites_shown(answer: str, hits: Sequence[Hit]) -> bool:
  """Whether the "Fonte:" line, when present, names a shown law (and article, if given)."""
  if SOURCE not in answer:
    return True
  citation = _citation(answer)
  if citation is None:
    return False
  name, number = citation
  return any(hit.name == name and number in (None, hit.number) for hit in hits)


def cites_asked(answer: str, question: str) -> bool:
  """Whether an answer to a question naming a law or article by number cites exactly those.

  "O que diz o art. 22 da Lei 11.340?" must not be answered with art. 46 of that law, nor
  with another law's article.
  """
  law, numbers = named_law(question), named_articles(question)
  if law is None and not numbers:
    return True
  citation = _citation(answer)
  if citation is None:
    return False
  name, number = citation
  return (law is None or name.startswith(law.prefix)) and (not numbers or number in numbers)


def grounded(answer: str, hits: Sequence[Hit], question: str = "") -> bool:
  """Whether the answer may be shown as it is: a deferral, or a cited excerpt of `hits`."""
  if answer.strip() == NOT_FOUND:
    return True
  return (
    coverage(answer, hits) >= GROUNDED_SHARE
    and cites_shown(answer, hits)
    and cites_asked(answer, question)
  )


def checked(answer: str, hits: Sequence[Hit], question: str = "") -> str:
  """The answer when grounded, else NOT_FOUND."""
  return answer if grounded(answer, hits, question) else NOT_FOUND
