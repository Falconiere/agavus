"""Tests for the CI commit-range gate (temporary git repos, no network)."""

import subprocess
from pathlib import Path

import pytest

from check_commit_range import USAGE_EXIT, ZERO_SHA, main, revisions


def _git(cwd: Path, *args: str) -> str:
  result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
  return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
  _git(tmp_path, "init", "-q")
  _git(tmp_path, "config", "user.email", "test@example.com")
  _git(tmp_path, "config", "user.name", "test")
  return tmp_path


def _commit(repo: Path, message: str) -> str:
  _git(repo, "commit", "-q", "--allow-empty", "-m", message)
  return _git(repo, "rev-parse", "HEAD")


def _committed_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *messages: str) -> list[str]:
  repo = _repo(tmp_path)
  hashes = [_commit(repo, message) for message in messages]
  monkeypatch.chdir(repo)
  return hashes


def _range_outcome(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  second: str,
) -> tuple[int, str]:
  revisions_ = _committed_range(tmp_path, monkeypatch, "feat(corpus): add ingestion", second)
  code = main(revisions_)
  return code, capsys.readouterr().out


@pytest.mark.parametrize(
  ("second", "expected_code"),
  [("fix(decide): correct beta threshold", 0), ("updated stuff", 1)],
)
def test_range_outcomes(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  second: str,
  expected_code: int,
) -> None:
  code, output = _range_outcome(tmp_path, monkeypatch, capsys, second)
  assert code == expected_code
  if expected_code == 0:
    assert output == ""
  else:
    assert second in output


def test_zero_start_checks_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  head = _committed_range(tmp_path, monkeypatch, "chore(repo): initialize")[0]
  assert revisions(ZERO_SHA, head) == [head]


def test_usage_error() -> None:
  assert main([]) == USAGE_EXIT
