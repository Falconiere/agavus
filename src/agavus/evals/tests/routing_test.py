"""Routing metrics: unsafe decisions are counted, defers are not wrong."""

from agavus.evals.goldset import GoldItem
from agavus.evals.routing import evaluate_routing
from agavus.evals.run import evaluate
from agavus.publish.router_gate import router_problems

F = {"F": 0.98, "M": 0.01, "O": 0.01}
M = {"F": 0.01, "M": 0.98, "O": 0.01}
UNSURE = {"F": 0.4, "M": 0.4, "O": 0.2}
ITEMS = [GoldItem("a", "x", "M"), GoldItem("b", "y", "M"), GoldItem("c", "z", "F")]


def test_counts_and_unsafe() -> None:
  report = evaluate_routing(ITEMS, [F, M, UNSURE])
  assert (report.right, report.wrong, report.deferred) == (1, 1, 1)
  assert (report.m_hits, report.m_total) == (1, 2)
  assert report.unsafe == ("a",)


def test_gate_blocks_unsafe_router() -> None:
  gold = evaluate([M, M, F], ITEMS)
  assert router_problems(gold, evaluate_routing(ITEMS, [M, M, F])) == []
  assert router_problems(gold, evaluate_routing(ITEMS, [F, M, F]))
