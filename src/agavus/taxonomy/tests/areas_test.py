"""Tests for the area registry (tests folder)."""

from dataclasses import FrozenInstanceError

import pytest

from agavus.taxonomy.areas import AREAS, FAMILIA, AreaStatus, is_area, routing_areas


def test_registry_has_f_m_o() -> None:
  assert set(AREAS) == {"F", "M", "O"}
  assert AREAS["F"] is FAMILIA


def test_is_area() -> None:
  assert is_area("M")
  assert not is_area("T")


def test_routing_excludes_out_of_scope() -> None:
  codes = {spec.code for spec in routing_areas()}
  assert codes == {"F", "M"}


def test_specs_are_frozen() -> None:
  with pytest.raises(FrozenInstanceError):
    FAMILIA.name = "other"  # type: ignore[misc]


def test_current_areas_collecting() -> None:
  assert FAMILIA.status is AreaStatus.COLLECTING
  assert all(spec.tribunals == ("TJDFT",) for spec in routing_areas())
