"""Tests for the three-way rule (tests folder, hand-computed cases)."""

from agavus.decide.rule import decide, thresholds

TOLERANCE = 1e-9


def test_confident_m_accepts() -> None:
  decision = decide({"F": 0.02, "M": 0.97, "O": 0.01})
  assert decision.action == "accept-M"


def test_confident_o_rejects() -> None:
  decision = decide({"F": 0.02, "M": 0.01, "O": 0.97})
  assert decision.action == "reject"


def test_ambiguous_defers() -> None:
  decision = decide({"F": 0.40, "M": 0.55, "O": 0.05})
  assert decision.action == "defer"


def test_tie_prefers_defer() -> None:
  decision = decide({"F": 0.0, "M": 0.0, "O": 0.0})
  assert decision.action == "defer"


def test_thresholds_match_design() -> None:
  alpha_f, beta_f = thresholds("F")
  alpha_m, beta_m = thresholds("M")
  assert abs(alpha_f - 8 / 9) < TOLERANCE
  assert abs(beta_f - 0.25) < TOLERANCE
  assert abs(alpha_m - 6 / 7) < TOLERANCE
  assert abs(beta_m - 0.125) < TOLERANCE


def test_expected_losses_auditable() -> None:
  decision = decide({"F": 0.40, "M": 0.55, "O": 0.05})
  assert abs(decision.expected_losses["accept-M"] - 2.8) < TOLERANCE
  assert decision.expected_losses["defer"] == 1.0
