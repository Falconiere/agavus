"""Tests for the TPU mapping (co-located; table monkeypatched, never guessed)."""

import pytest

from agavus.taxonomy import tpu


def test_unknown_code_is_none() -> None:
  assert tpu.map_code("999999") is None
  assert tpu.map_codes(("999999",)) == set()


def test_verified_codes_map(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setitem(tpu.CODE_TO_AREA, "333", "M")
  monkeypatch.setitem(tpu.CODE_TO_AREA, "444", "F")
  assert tpu.map_code("333") == "M"
  assert tpu.map_codes(("333", "444", "999")) == {"M", "F"}


def test_version_present() -> None:
  assert tpu.TPU_VERSION.startswith("tpu-v")
