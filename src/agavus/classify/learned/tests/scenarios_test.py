"""Black-box relation-corpus tests derived from the approved experiment plan."""

from collections import Counter

from agavus.classify.learned.partitions import partition
from agavus.classify.learned.scenarios import relation_samples

FAMILIES = 120
PAIRS_PER_FAMILY = 6
OUTCOMES_PER_LABEL = 2
SAMPLES_PER_LABEL = FAMILIES * OUTCOMES_PER_LABEL


def test_relation_scenarios_are_balanced_grouped_and_provenanced() -> None:
  samples = relation_samples()
  expected_categories = {
    "direct_negation",
    "double_negation",
    "passive",
    "quotation",
    "conditional",
    "mixed_requests",
  }
  assert len(samples) == FAMILIES * PAIRS_PER_FAMILY
  assert Counter(sample.label for sample in samples) == {
    "support": SAMPLES_PER_LABEL,
    "contradiction": SAMPLES_PER_LABEL,
    "unknown": SAMPLES_PER_LABEL,
  }
  assert Counter(sample.category for sample in samples) == dict.fromkeys(
    expected_categories, FAMILIES
  )
  assert set(Counter(sample.group for sample in samples).values()) == {PAIRS_PER_FAMILY}
  assert len({sample.group for sample in samples}) == FAMILIES
  assert all(
    sample.task == "relation" and sample.statement and sample.origin_ref and sample.label_source
    for sample in samples
  )


def test_each_relation_family_has_two_examples_of_every_outcome() -> None:
  grouped: dict[str, Counter[str]] = {}
  for sample in relation_samples():
    grouped.setdefault(sample.group, Counter())[sample.label] += 1
  assert all(
    counts == dict.fromkeys(("support", "contradiction", "unknown"), OUTCOMES_PER_LABEL)
    for counts in grouped.values()
  )


def test_relation_templates_are_not_shared_across_partitions() -> None:
  samples = relation_samples()
  parts = partition(samples)
  location = {index: name for name, rows in vars(parts).items() for index in rows}
  templates: dict[str, set[str]] = {}
  for index, sample in enumerate(samples):
    templates.setdefault(sample.template, set()).add(location[index])
  assert all(len(locations) == 1 for locations in templates.values())
