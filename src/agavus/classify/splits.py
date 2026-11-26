"""Case-grouped training, calibration and final evaluation partitions."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

MIN_GROUPS = 3
HOLDOUT_EVERY = 5


@dataclass(frozen=True)
class TrainingData:
  """Reviewed examples with the provenance needed to prevent case leakage."""

  texts: tuple[str, ...]
  labels: tuple[str, ...]
  groups: tuple[str, ...]
  dates: tuple[date | None, ...]


@dataclass(frozen=True)
class TrainingSplit:
  """Disjoint row indexes; calibration and final holdout have separate jobs."""

  train: tuple[int, ...]
  calibration: tuple[int, ...]
  holdout: tuple[int, ...]


def _root(parents: list[int], index: int) -> int:
  while parents[index] != index:
    parents[index] = parents[parents[index]]
    index = parents[index]
  return index


def _group_rows(texts: Sequence[str], groups: Sequence[str]) -> list[tuple[int, ...]]:
  """Join case identity and duplicate text links transitively."""
  parents = list(range(len(texts)))
  seen: dict[tuple[str, str], int] = {}
  for index, (text, group) in enumerate(zip(texts, groups, strict=True)):
    for key in (("case", group), ("text", " ".join(text.casefold().split()))):
      previous = seen.setdefault(key, index)
      parents[_root(parents, index)] = _root(parents, previous)
  members: dict[int, list[int]] = {}
  for index in range(len(texts)):
    members.setdefault(_root(parents, index), []).append(index)
  return [tuple(rows) for rows in members.values()]


def split_examples(
  texts: Sequence[str],
  labels: Sequence[str],
  *,
  groups: Sequence[str] | None = None,
  dates: Sequence[date | None] | None = None,
) -> TrainingSplit:
  """Reserve the newest groups per class for evaluation, then calibration."""
  identities = tuple(groups) if groups is not None else tuple(texts)
  published = tuple(dates) if dates is not None else (None,) * len(texts)
  if not texts or not len(texts) == len(labels) == len(identities) == len(published):
    raise ValueError("need nonempty aligned texts, labels, groups and dates")
  by_label: dict[str, list[tuple[date, str, tuple[int, ...]]]] = {}
  for rows in _group_rows(texts, identities):
    areas = {labels[index] for index in rows}
    if len(areas) != 1:
      raise ValueError("conflicting labels within a case or duplicate-text group")
    latest = max(published[index] or date.min for index in rows)
    identity = repr(sorted({identities[index] for index in rows}))
    digest = hashlib.sha256(identity.encode()).hexdigest()
    by_label.setdefault(labels[rows[0]], []).append((latest, digest, rows))
  train: list[int] = []
  calibration: list[int] = []
  holdout: list[int] = []
  for label in sorted(by_label):
    ordered = sorted(by_label[label])
    dated = [group for group in ordered if group[0] != date.min]
    candidates = dated or ordered
    if len(candidates) < MIN_GROUPS:
      raise ValueError("need at least three independent groups per class")
    reserve = max(1, len(candidates) // HOLDOUT_EVERY)
    for destination, partition in (
      (train, ordered[: -2 * reserve]),
      (calibration, ordered[-2 * reserve : -reserve]),
      (holdout, ordered[-reserve:]),
    ):
      destination.extend(index for _, _, rows in partition for index in rows)
  return TrainingSplit(tuple(sorted(train)), tuple(sorted(calibration)), tuple(sorted(holdout)))
