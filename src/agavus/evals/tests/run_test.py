"""Tests for gold-set evaluation (tests folder, hand-built probabilities)."""

from agavus.evals.goldset import GoldItem
from agavus.evals.run import evaluate

EXPECTED_ACCEPTED = 2
HALF = 0.5
CONFIDENT_M = {"F": 0.01, "M": 0.98, "O": 0.01}
CONFIDENT_F = {"F": 0.98, "M": 0.01, "O": 0.01}
CONFIDENT_O = {"F": 0.01, "M": 0.01, "O": 0.98}
AMBIGUOUS_MF = {"F": 0.40, "M": 0.55, "O": 0.05}
WEAK_O = {"F": 0.30, "M": 0.30, "O": 0.40}

GOLD = (
  GoldItem("g1", "texto m", "M"),
  GoldItem("g2", "texto f", "F"),
  GoldItem("g3", "texto o", "O"),
)


def test_perfect_confident_predictions() -> None:
  report = evaluate([CONFIDENT_M, CONFIDENT_F, CONFIDENT_O], GOLD)
  assert report.accuracy == 1.0
  assert report.accepted == EXPECTED_ACCEPTED
  assert report.accepted_correct == EXPECTED_ACCEPTED
  assert report.rejected == 1
  assert report.precision_on_accepted == 1.0
  assert report.m_recall == 1.0


def test_ambiguous_defers_and_lowers_coverage() -> None:
  report = evaluate([AMBIGUOUS_MF, CONFIDENT_F, WEAK_O], GOLD)
  assert report.deferred >= 1
  assert report.coverage < 1.0


def test_wrong_accept_hurts_precision() -> None:
  report = evaluate([CONFIDENT_F, CONFIDENT_F, CONFIDENT_O], GOLD)
  assert report.precision_on_accepted == HALF
  assert report.m_recall == 0.0
  assert report.accepted_correct == 1
  assert report.accepted == EXPECTED_ACCEPTED
