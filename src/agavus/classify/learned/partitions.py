"""Four frozen partitions, joining case, duplicate and template links transitively."""

import hashlib
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from agavus.classify.learned.types import RELATIONS, Sample
from agavus.classify.splits import _group_rows
from agavus.taxonomy.areas import AREAS

MIN_GROUPS = 4


@dataclass(frozen=True)
class Partitions:
  """Row indexes with distinct weight, checkpoint, temperature and scoring roles."""

  train: tuple[int, ...]
  validation: tuple[int, ...]
  calibration: tuple[int, ...]
  final: tuple[int, ...]


def _ordered_groups(
  samples: Sequence[Sample],
) -> dict[str, list[tuple[date, str, tuple[int, ...]]]]:
  grouped: dict[str, list[tuple[date, str, tuple[int, ...]]]] = {}
  families = _group_rows(
    [unicodedata.normalize("NFKC", s.text) for s in samples], [s.group for s in samples]
  )
  components = {index: str(root) for root, rows in enumerate(families) for index in rows}
  linked = _group_rows(
    [s.template or f"unique:{s.id}" for s in samples],
    [components[index] for index in range(len(samples))],
  )
  for rows in linked:
    members = [samples[index] for index in rows]
    tasks = {s.task for s in members}
    labels = {s.label for s in members}
    if len(tasks) != 1 or (tasks == {"area"} and len(labels) != 1):
      raise ValueError("conflicting labels within a case or duplicate-text group")
    if any(s.auxiliary for s in members):
      if not all(s.auxiliary for s in members):
        raise ValueError("auxiliary data duplicates evaluation candidates")
      key = "auxiliary"
    else:
      key = "relation" if tasks == {"relation"} else f"area:{members[0].label}"
    latest = max(s.published_at or date.min for s in members)
    identity = repr(sorted({s.group for s in members}))
    digest = hashlib.sha256(identity.encode()).hexdigest()
    grouped.setdefault(key, []).append((latest, digest, rows))
  return grouped


def partition(samples: Sequence[Sample]) -> Partitions:
  """Split independent groups per class; auxiliary rows cannot displace dated cases."""
  if not samples or len({s.id for s in samples}) != len(samples):
    raise ValueError("need nonempty samples with unique ids")
  if any(not s.origin_ref or not s.label_source or not s.text.strip() for s in samples):
    raise ValueError("text and provenance are mandatory")
  if any(s.label not in (RELATIONS if s.task == "relation" else AREAS) for s in samples):
    raise ValueError("unregistered task label")
  buckets: list[list[int]] = [[], [], [], []]
  for key, groups in _ordered_groups(samples).items():
    ordered = sorted(groups)
    if key == "auxiliary":
      buckets[0].extend(i for _, _, rows in ordered for i in rows)
      continue
    candidates = list(filter(lambda group: group[0] > date.min, ordered))
    has_dates = bool(candidates)
    if not has_dates:
      candidates = ordered
    if len(candidates) < MIN_GROUPS:
      raise ValueError(f"need at least four independent groups per class: {key}")
    if has_dates:
      buckets[0].extend(i for day, _, rows in ordered if day == date.min for i in rows)
    count = len(candidates)
    n_validation, n_calibration, n_final = (max(1, int(count * f)) for f in (0.15, 0.10, 0.15))
    bounds = (
      0,
      count - n_validation - n_calibration - n_final,
      count - n_calibration - n_final,
      count - n_final,
      count,
    )
    for bucket, start, end in zip(buckets, bounds, bounds[1:], strict=False):
      bucket.extend(i for _, _, rows in candidates[start:end] for i in rows)
  result = Partitions(*(tuple(sorted(bucket)) for bucket in buckets))
  _validate_classes(samples, result)
  return result


def _validate_classes(samples: Sequence[Sample], parts: Partitions) -> None:
  expected = {(s.task, s.label) for s in samples if not s.auxiliary}
  for rows in (parts.train, parts.validation, parts.calibration, parts.final):
    actual = {(samples[i].task, samples[i].label) for i in rows if not samples[i].auxiliary}
    if actual != expected:
      raise ValueError("need an independent group per partition and class")
