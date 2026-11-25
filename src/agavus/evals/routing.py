"""Routing evaluation on what the user experiences: the decision."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agavus.decide.losses import accept_action
from agavus.decide.rule import decide
from agavus.evals.goldset import GoldItem
from agavus.taxonomy.areas import MARIA_DA_PENHA, OUT_OF_SCOPE


@dataclass(frozen=True)
class RoutingReport:
  """Decision counts; unsafe lists M questions accepted elsewhere or rejected."""

  right: int
  wrong: int
  deferred: int
  m_hits: int
  m_total: int
  unsafe: tuple[str, ...]


def evaluate_routing(
  items: Sequence[GoldItem], probabilities: Sequence[Mapping[str, float]]
) -> RoutingReport:
  """Compare every decision with the right one."""
  actions = [decide(mapping).action for mapping in probabilities]
  pairs = list(zip(items, actions, strict=True))
  m_code = MARIA_DA_PENHA.code
  hits = sum(
    item.expected == m_code and max(mapping, key=lambda area: mapping[area]) == m_code
    for item, mapping in zip(items, probabilities, strict=True)
  )
  return RoutingReport(
    right=sum(action == _right_action(item) for item, action in pairs),
    wrong=sum(action not in ("defer", _right_action(item)) for item, action in pairs),
    deferred=actions.count("defer"),
    m_hits=hits,
    m_total=sum(item.expected == m_code for item in items),
    unsafe=tuple(
      item.id
      for item, action in pairs
      if item.expected == m_code and action not in ("defer", accept_action(m_code))
    ),
  )


def _right_action(item: GoldItem) -> str:
  return "reject" if item.expected == OUT_OF_SCOPE.code else accept_action(item.expected)
