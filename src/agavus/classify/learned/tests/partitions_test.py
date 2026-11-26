"""Black-box partition tests derived from the approved experiment plan."""

from collections import Counter
from dataclasses import replace
from datetime import date, timedelta

import pytest

from agavus.classify.learned.partitions import partition
from agavus.classify.learned.types import Sample

from .conftest import partition_locations

AREA_SAMPLES = 20
FAMILY_MEMBERS = 2


def _area_samples(*, auxiliary: bool = False) -> tuple[Sample, ...]:
  labels = ("F", "M", "O")
  return tuple(
    Sample(
      id=f"{label}-{index}{'-aux' if auxiliary else ''}",
      text=f"texto {label} {index}{' auxiliar' if auxiliary else ''}",
      label=label,
      group=f"case-{label}-{index}{'-aux' if auxiliary else ''}",
      origin_ref=f"Acórdão {label}-{index}",
      label_source="tpu",
      published_at=None if auxiliary else date(2020, 1, 1) + timedelta(days=index),
      auxiliary=auxiliary,
    )
    for label in labels
    for index in range(AREA_SAMPLES)
  )


def _dated(samples: tuple[Sample, ...], rows: tuple[int, ...]) -> list[date]:
  values: list[date] = []
  for index in rows:
    sample = samples[index]
    if not sample.auxiliary and sample.published_at is not None:
      values.append(sample.published_at)
  return values


def test_partition_isolates_transitive_duplicate_and_case_groups() -> None:
  samples = list(_area_samples())
  samples[0] = Sample(
    id="F-0",
    text="decisão repetida",
    label="F",
    group="case-f-one",
    origin_ref="Acórdão F-0",
    label_source="tpu",
    published_at=date(2020, 1, 1),
  )
  samples[1] = Sample(
    id="F-1",
    text="decisão repetida",
    label="F",
    group="case-f-two",
    origin_ref="Acórdão F-1",
    label_source="tpu",
    published_at=date(2020, 1, 2),
  )
  parts = partition(samples)
  locations = [rows for rows in vars(parts).values() if 0 in rows or 1 in rows]
  assert len(locations) == 1
  assert set().union(*vars(parts).values()) == set(range(len(samples)))
  for rows in vars(parts).values():
    assert {samples[index].label for index in rows} == {"F", "M", "O"}


def test_partition_orders_dated_cases_and_keeps_auxiliaries_in_training() -> None:
  samples = _area_samples() + _area_samples(auxiliary=True)
  parts = partition(samples)
  assert max(_dated(samples, parts.train)) <= min(_dated(samples, parts.validation))
  assert max(_dated(samples, parts.validation)) <= min(_dated(samples, parts.calibration))
  assert max(_dated(samples, parts.calibration)) <= min(_dated(samples, parts.final))
  for rows, expected in zip(vars(parts).values(), (12, 3, 2, 3), strict=True):
    assert Counter(samples[index].label for index in rows if not samples[index].auxiliary) == {
      "F": expected,
      "M": expected,
      "O": expected,
    }
  assert all(samples[index].auxiliary for index in parts.train if index >= 3 * AREA_SAMPLES)
  held_out = parts.validation + parts.calibration + parts.final
  assert all(not samples[index].auxiliary for index in held_out)


def test_partition_unions_family_and_template_chains_with_missing_template_members() -> None:
  labels = ("support", "support", "contradiction", "contradiction", "unknown", "unknown")
  samples = tuple(
    Sample(
      id=f"{family}-{offset}",
      text=f"passage {family}-{offset}",
      label=label,
      group=f"family-{family}" if offset < FAMILY_MEMBERS else f"family-{family}-alias",
      origin_ref=f"synthetic:{family}-{offset}",
      label_source="synthetic",
      statement="statement",
      template="" if offset == 0 else f"template-{family}",
    )
    for family in range(AREA_SAMPLES)
    for offset, label in enumerate(labels)
  )
  parts = partition(samples)
  location = partition_locations(parts)
  assert all(
    len({location[family * len(labels) + offset] for offset in range(len(labels))}) == 1
    for family in range(AREA_SAMPLES)
  )


def test_partition_rejects_an_unregistered_area_label() -> None:
  samples = list(_area_samples())
  samples[0] = replace(samples[0], label="X")
  with pytest.raises(ValueError, match="unregistered"):
    partition(samples)
