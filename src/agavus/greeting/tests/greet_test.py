"""Tests for the greeting domain (in the domain tests folder)."""

from agavus.greeting.greet import greet


def test_greet_world() -> None:
  assert greet("world") == "Hello, world!"


def test_greet_named() -> None:
  assert greet("Brasil") == "Hello, Brasil!"
