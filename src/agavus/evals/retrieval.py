"""Retrieval evaluation: is the right article found, and is it first?"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agavus.evals.retrievalset import RetrievalItem

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
