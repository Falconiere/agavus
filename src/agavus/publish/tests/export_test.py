"""Tests for release gating and staging (tests folder, synthetic models)."""

from pathlib import Path

import pytest

from agavus.classify.pipeline import train_classifier
from agavus.classify.seed import SEED_QUERIES
from agavus.classify.store import save_model
from agavus.config.settings import Settings
from agavus.corpus.httpfixtures import sample_settings
from agavus.evals.run import EvalReport
from agavus.publish.export import gate, stage_release


def _report(m_recall: float = 1.0, precision: float = 1.0, accepted: int = 3) -> EvalReport:
  return EvalReport(
    accuracy=1.0,
    macro_f1=1.0,
    accepted=accepted,
    accepted_correct=accepted,
    deferred=2,
    rejected=2,
    precision_on_accepted=precision,
    coverage=0.6,
    m_recall=m_recall,
  )


def test_gate_passes_good_report() -> None:
  assert gate(_report()) == []


def test_gate_blocks_unsafe_reports() -> None:
  assert gate(_report(m_recall=0.5))
  assert gate(_report(precision=0.9))
  assert gate(_report(accepted=0))


def _settings(tmp_path: Path) -> Settings:
  return sample_settings(tmp_path)


def test_stage_release_builds_card(tmp_path: Path) -> None:
  settings = _settings(tmp_path)
  texts = [text for text, _ in SEED_QUERIES]
  labels = [label for _, label in SEED_QUERIES]
  classifier, report = train_classifier(texts, labels)
  save_model(tmp_path / "models" / "models.db", "classifier", classifier, report)
  staging = tmp_path / "staging"
  stage_release(settings, staging)
  assert (staging / "models.db").exists()
  card = (staging / "README.md").read_text(encoding="utf-8")
  assert "agavus-classifier" in card
  assert "M-recall" in card
  assert (staging / "eval_report.json").exists()


def test_stage_release_refuses_without_model(tmp_path: Path) -> None:
  settings = _settings(tmp_path)
  with pytest.raises(KeyError):
    stage_release(settings, tmp_path / "staging")
