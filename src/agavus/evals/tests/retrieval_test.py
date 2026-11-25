"""Retrieval metrics and question-set integrity."""

import pytest

from agavus.evals.retrieval import evaluate_retrieval, rank_of
from agavus.evals.retrievalset import DEV, HIDDEN, RetrievalItem

SET_SIZE = 40


def test_rank_of() -> None:
  assert rank_of(["a", "b"], ["b", "z"]) == len(["a", "b"])
  assert rank_of(["a"], ["z"]) == 0


def test_report_counts() -> None:
  items = [RetrievalItem("1", "x", ("a",)), RetrievalItem("2", "y", ("z",))]
  report = evaluate_retrieval(items, lambda _: ["a", "b"])
  assert (report.ranks, report.first, report.in_top_k) == ((1, 0), 1, 1)
  assert report.mrr == pytest.approx(0.5)


def test_sets_are_complete_and_disjoint() -> None:
  assert len(DEV) == len(HIDDEN) == SET_SIZE
  assert all(item.relevant for item in DEV + HIDDEN)
  assert len({item.text for item in DEV + HIDDEN}) == 2 * SET_SIZE
  assert len({item.id for item in DEV + HIDDEN}) == 2 * SET_SIZE
