"""Tests for the Space app (tests folder; seeds-only model, no network, word search)."""

from pathlib import Path

import pytest

from agavus.classify.pipeline import train_classifier, training_set
from agavus.classify.store import save_model
from agavus.corpus.storage import CorpusStorage
from space import app
from space.app import answer_question


def test_empty_question_prompts() -> None:
  header, body, trace = answer_question("   ")
  assert header == ""
  assert "Digite" in body
  assert trace == ""


def test_answer_flow_with_seeds_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  storage = CorpusStorage(tmp_path)
  texts, targets = training_set(storage)
  classifier, report = train_classifier(texts, targets)
  save_model(tmp_path / "models" / "models.db", "classifier", classifier, report)
  monkeypatch.setattr(app, "prepare_encoder", lambda data_dir: None)
  monkeypatch.setenv("AGAVUS_DATA_DIR", str(tmp_path))
  header, body, trace = answer_question("ele descumpriu a medida protetiva")
  assert "Decisão" in header
  assert body
  assert "probabilidades" in trace
