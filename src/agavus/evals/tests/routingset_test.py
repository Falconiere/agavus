"""Routing set integrity: size, uniqueness, firewall against training seeds."""

from agavus.classify.seed import SEED_QUERIES
from agavus.classify.seed_everyday import EVERYDAY_QUERIES
from agavus.evals.goldset import ROUTING_V1
from agavus.evals.routingset import DEV, HIDDEN
from agavus.taxonomy.areas import is_area

SET_SIZE = 50


def test_sets_are_complete_and_disjoint() -> None:
  assert len(DEV) == len(HIDDEN) == SET_SIZE
  items = (*DEV, *HIDDEN, *ROUTING_V1)
  assert len({item.text.casefold() for item in items}) == len(items)
  assert all(is_area(item.expected) for item in items)


def test_seeds_never_contain_evaluation_questions() -> None:
  seeds = {text.casefold().strip(" ?.") for text, _ in SEED_QUERIES + EVERYDAY_QUERIES}
  assert not seeds & {item.text.casefold().strip(" ?.") for item in (*DEV, *HIDDEN, *ROUTING_V1)}
