"""Tests for the Space app (tests folder; synthetic router, no network)."""

from pathlib import Path

import pytest

from agavus.answer.compose import UNAVAILABLE_MESSAGE
from agavus.classify.fixtures import UnsureRouter
from space import app
from space.app import answer_question


def test_empty_question_prompts() -> None:
  header, body, trace = answer_question("   ")
  assert header == ""
  assert "Digite" in body
  assert trace == ""


def _without_bootstrap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, encoder: object) -> None:
  monkeypatch.setattr(app, "_bootstrap", lambda data_dir: tmp_path / "models.db")
  monkeypatch.setattr(app, "_encoder", lambda data_dir: encoder)
  monkeypatch.setattr(app, "ensure_router", lambda path, encoder: UnsureRouter())
  monkeypatch.setenv("AGAVUS_DATA_DIR", str(tmp_path))


def test_without_encoder_every_question_is_deferred(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _without_bootstrap(monkeypatch, tmp_path, None)
  header, body, trace = answer_question("ele descumpriu a medida protetiva")
  assert "defer" in header
  assert UNAVAILABLE_MESSAGE in body
  assert trace == ""


def test_answer_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  _without_bootstrap(monkeypatch, tmp_path, object())
  header, body, trace = answer_question("ele descumpriu a medida protetiva")
  assert "defer" in header
  assert body
  assert "probabilidades" in trace
