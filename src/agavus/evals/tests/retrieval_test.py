"""Retrieval metrics and question-set integrity."""

import pytest

from agavus.answer.compose import Answer
from agavus.evals.passageset import FRESH
from agavus.evals.retrieval import CitationReport, evaluate_citations, evaluate_retrieval, rank_of
from agavus.evals.retrievalset import DEV, HIDDEN, RetrievalItem
from agavus.taxonomy.areas import FAMILIA, MARIA_DA_PENHA

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


def test_fresh_set_is_disjoint_and_has_paired_passages() -> None:
  assert len(FRESH) == SET_SIZE
  rows = FRESH + DEV + HIDDEN
  assert len({item.id for item in rows}) == len({item.text for item in rows}) == len(rows)
  for area in (FAMILIA.code, MARIA_DA_PENHA.code):
    items = [item for item in FRESH if item.area == area]
    assert any(not item.relevant for item in items)
    assert any(item.passages for item in items)
  assert all(len(item.relevant) == len(item.passages) for item in FRESH)


def test_score_only_the_displayed_primary_excerpt() -> None:
  outcomes = (
    Answer("accept-M", "a:\n\nprova\tlegal", ("a",)),
    Answer("accept-M", "Fatos: prova legal\n\na:\noutra coisa\n\nContexto: prova legal", ("a",)),
    Answer("accept-F", "b:\n\nprova legal\n\nContexto: a", ("b", "a")),
    Answer("accept-M", "b:\n\nprova legal\n\nContexto: a", ("a", "b")),
    Answer("reject", "", ()),
    Answer("defer", "", ()),
  )
  items = [
    RetrievalItem("", str(i), ("a",), MARIA_DA_PENHA.code, ("prova legal",))
    for i in range(len(outcomes))
  ]
  assert evaluate_citations(items, lambda q: outcomes[int(q)]) == CitationReport(2, 2, 1, 1, 1, 2)
