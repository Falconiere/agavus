"""Tests for seed queries integrity (tests folder; gold firewall)."""

from agavus.classify.seed import SEED_QUERIES
from agavus.evals.goldset import ROUTING_V1
from agavus.taxonomy.areas import is_area


def test_labels_are_registered_areas() -> None:
  assert all(is_area(label) for _, label in SEED_QUERIES)


def test_all_areas_covered() -> None:
  assert {label for _, label in SEED_QUERIES} == {"F", "M", "O"}


def test_gold_firewall_no_exact_overlap() -> None:
  gold_texts = {item.text.casefold() for item in ROUTING_V1}
  seed_texts = {text.casefold() for text, _ in SEED_QUERIES}
  assert gold_texts.isdisjoint(seed_texts)


def test_no_duplicate_seeds() -> None:
  texts = [text for text, _ in SEED_QUERIES]
  assert len(set(texts)) == len(texts)
