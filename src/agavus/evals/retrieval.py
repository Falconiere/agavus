"""Retrieval evaluation: is the right article found, and is it first?"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agavus.answer.compose import DISCLAIMER, Answer
from agavus.decide.losses import accept_action
from agavus.evals.retrievalset import RetrievalItem
from agavus.taxonomy.areas import MARIA_DA_PENHA

TOP_K = 5


@dataclass(frozen=True)
class RetrievalReport:
  """Counts over one question set (rank 0 = not found)."""

  ranks: tuple[int, ...]
  first: int
  in_top_k: int
  mrr: float


def rank_of(refs: Sequence[str], relevant: Sequence[str]) -> int:
  """1-based position of the first correct article; 0 when absent."""
  return next((place for place, ref in enumerate(refs, 1) if ref in relevant), 0)


def evaluate_retrieval(
  items: Sequence[RetrievalItem], retrieve: Callable[[str], Sequence[str]]
) -> RetrievalReport:
  """Rank of the first correct article for every question."""
  ranks = tuple(rank_of(retrieve(item.text), item.relevant) for item in items)
  reciprocal = [1 / rank if rank else 0.0 for rank in ranks]
  return RetrievalReport(
    ranks=ranks,
    first=sum(rank == 1 for rank in ranks),
    in_top_k=sum(0 < rank <= TOP_K for rank in ranks),
    mrr=sum(reciprocal) / len(ranks) if ranks else 0.0,
  )


@dataclass(frozen=True)
class CitationReport:
  """Primary citation identity, displayed passage support and delivery outcomes."""

  correct: int
  wrong: int
  deferred: int
  rejected: int
  supported: int
  unsafe_m: int


def _supports(answer: Answer, item: RetrievalItem, primary: str) -> bool:
  """Score only the rendered excerpt or verified premise following its citation."""
  _, heading, body = answer.text.partition(f"{primary}:\n")
  quote = body.partition("\n\nContexto: ")[0].removesuffix(f"\n\n{DISCLAIMER}")
  return bool(heading) and any(
    ref == primary and passage in " ".join(quote.split())
    for ref, passage in zip(item.relevant, item.passages, strict=True)
  )


def evaluate_citations(
  items: Sequence[RetrievalItem], respond: Callable[[str], Answer]
) -> CitationReport:
  """Count each final answer once. Only annotated questions can earn passage credit."""
  correct = wrong = deferred = rejected = supported = unsafe_m = 0
  for item in items:
    answer = respond(item.text)
    if item.area == MARIA_DA_PENHA.code and answer.decision not in (
      "defer",
      accept_action(item.area),
    ):
      unsafe_m += 1
    if answer.decision == "defer":
      deferred += 1
    elif answer.decision == "reject":
      rejected += 1
    elif answer.sources and answer.sources[0] in item.relevant:
      correct += 1
      if item.passages and _supports(answer, item, answer.sources[0]):
        supported += 1
    else:
      wrong += 1
  return CitationReport(correct, wrong, deferred, rejected, supported, unsafe_m)
