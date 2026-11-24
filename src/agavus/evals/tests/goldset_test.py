"""Tests for the routing gold set integrity (co-located)."""

from agavus.evals.goldset import GOLD_VERSION, ROUTING_V1
from agavus.taxonomy.areas import is_area


def test_gold_set_size() -> None:
  expected_items = 10
  assert len(ROUTING_V1) == expected_items
  assert GOLD_VERSION == "routing-v1"


def test_ids_unique() -> None:
  ids = [item.id for item in ROUTING_V1]
  assert len(set(ids)) == len(ids)


def test_expected_labels_are_registered_areas() -> None:
  assert all(is_area(item.expected) for item in ROUTING_V1)


def test_all_strata_present() -> None:
  labels = {item.expected for item in ROUTING_V1}
  assert labels == {"F", "M", "O"}
