"""Tests for the newest-to-oldest cursor."""

from agavus.corpus.cursor import WalkState, comes_before, decode, encode


def test_roundtrip_and_legacy_date_is_ignored() -> None:
  state = WalkState(("20240510120000", "abc"), ("20261124101010", "zzz"), False)
  assert decode(encode(state)) == state
  assert decode("2024-05-10") == WalkState(None, None, False)
  assert decode(None) == WalkState(None, None, False)


def test_comes_before_is_newer_first() -> None:
  assert comes_before(("20240601120000", "a"), ("20240501120000", "a"))
  assert not comes_before(("20240501120000", "a"), ("20240601120000", "a"))
  assert comes_before(("20240510120000", "b"), ("20240510120000", "a"))
  assert not comes_before(("20240510120000", "a"), ("20240510120000", "a"))
  assert comes_before(("2024-06-01T00:00:00.000Z", "a"), ("2024-05-01T00:00:00.000Z", "b"))
