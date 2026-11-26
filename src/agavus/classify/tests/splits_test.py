"""Tests for deterministic, leakage-safe classifier data splits."""

from datetime import date

import pytest

from agavus.classify.splits import TrainingSplit, split_examples

UNDATED_PER_CLASS = 8
DATED_PER_CLASS = 5
TRAIN_PER_CLASS = UNDATED_PER_CLASS + DATED_PER_CLASS - 2
ExampleRows = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[date | None, ...]]


def _dated_examples(per_class: int, undated: int = 0) -> ExampleRows:
  labels = tuple(label for label in "FM" for _ in range(per_class))
  groups = tuple(f"{label}-{index}" for label in "FM" for index in range(per_class))
  texts = tuple(f"texto {group}" for group in groups)
  dates = tuple(
    None if index < undated else date(2024, 1, index - undated + 1)
    for _ in "FM"
    for index in range(per_class)
  )
  return texts, labels, groups, dates


def _split_dated_examples(
  per_class: int, undated: int = 0
) -> tuple[TrainingSplit, tuple[str, ...]]:
  texts, labels, groups, dates = _dated_examples(per_class, undated)
  return split_examples(texts, labels, groups=groups, dates=dates), groups


def test_split_keeps_duplicate_texts_and_explicit_groups_together() -> None:
  """Catch leakage when equivalent text links otherwise separate case groups."""
  texts = (
    "Guarda compartilhada",
    "  guarda   compartilhada  ",
    "família três",
    "família quatro",
    "violência um",
    "violência dois",
    "violência três",
  )
  labels = ("F", "F", "F", "F", "M", "M", "M")
  groups = ("f1", "f2", "f3", "f4", "m1", "m2", "m3")
  dates = tuple(date(2024, 1, day) for day in range(1, len(texts) + 1))

  first = split_examples(texts, labels, groups=groups, dates=dates)
  second = split_examples(texts, labels, groups=groups, dates=dates)

  assert first == second
  memberships = {
    index: split_name
    for split_name, indexes in (
      ("train", first.train),
      ("calibration", first.calibration),
      ("holdout", first.holdout),
    )
    for index in indexes
  }
  assert memberships[0] == memberships[1]
  assert set(memberships) == set(range(len(texts)))


def test_split_rejects_misaligned_metadata() -> None:
  """Catch metadata that could silently associate examples with wrong cases."""
  with pytest.raises(ValueError):
    split_examples(("a", "b", "c"), ("F", "F", "F"), groups=("one", "two"))


def test_split_reserves_newest_groups_for_holdout_then_calibration() -> None:
  """Catch chronology reversal that lets the newest cases train the model."""
  split, groups = _split_dated_examples(5)

  assert {groups[index] for index in split.holdout} == {"F-4", "M-4"}
  assert {groups[index] for index in split.calibration} == {"F-3", "M-3"}


def test_split_keeps_undated_groups_in_train_when_cases_have_dates() -> None:
  """Catch date-less auxiliary text consuming real-case evaluation capacity."""
  split, groups = _split_dated_examples(UNDATED_PER_CLASS + DATED_PER_CLASS, UNDATED_PER_CLASS)

  assert {groups[index] for index in split.holdout} == {"F-12", "M-12"}
  assert {groups[index] for index in split.calibration} == {"F-11", "M-11"}
  assert len(split.train) == TRAIN_PER_CLASS * len("FM")


@pytest.mark.parametrize(
  ("labels", "groups"),
  [
    (("F", "F"), ("one", "two")),
    (("F", "M", "F", "F"), ("one", "one", "two", "three")),
  ],
)
def test_split_rejects_insufficient_or_conflicting_group_labels(
  labels: tuple[str, ...], groups: tuple[str, ...]
) -> None:
  """Catch a split with too little group diversity or contradictory case labels."""
  texts = tuple(f"texto {index}" for index in range(len(labels)))

  with pytest.raises(ValueError):
    split_examples(texts, labels, groups=groups)
