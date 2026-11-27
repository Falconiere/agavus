"""Lab reporting: steps chosen on validation, losses returned (synthetic text)."""

import pytest

from agavus.language_model.bigram import train_bigram
from agavus.language_model.data import Split
from agavus.language_model.report import chosen_steps, report


def test_lowest_validation_wins(capsys: pytest.CaptureFixture[str]) -> None:
  best = 200
  history = [(100, 2.0, 1.9), (best, 1.5, 1.4), (300, 1.0, 1.6)]
  assert chosen_steps(history, shown=2) == best
  assert "steps chosen on validation: 200" in capsys.readouterr().out


def test_report_returns_what_it_prints(capsys: pytest.CaptureFixture[str]) -> None:
  split = Split(("guarda guarda",), ("a guarda",))
  training, holdout, sample = report(train_bigram(split.train), split)
  printed = capsys.readouterr().out
  assert f"{training:.4f}" in printed
  assert f"{holdout:.4f}" in printed
  assert sample in printed
