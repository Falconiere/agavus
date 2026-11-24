"""Validate every commit message in a git range (CI; docs/commits.md).

The pre-commit hook checks one message locally; CI checks the whole push or
pull-request range, so the convention holds even without installed hooks.
"""

import subprocess
import sys
from collections.abc import Sequence

from check_commit_msg import validate

ZERO_SHA = "0" * 40
EXPECTED_ARGS = 2
USAGE_EXIT = 2


def _git(*args: str) -> str:
  result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
  return result.stdout


def revisions(start: str, end: str) -> list[str]:
  """Commits introduced by start..end; an empty/zero start checks end only."""
  if not start or start == ZERO_SHA:
    return [_git("rev-parse", end).strip()]
  return _git("rev-list", f"{start}..{end}").split()


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point: check <start> <end> commit messages; 1 on violations."""
  args = list(sys.argv[1:] if argv is None else argv)
  if len(args) != EXPECTED_ARGS:
    print("usage: check_commit_range.py <start-sha> <end-sha>")
    return USAGE_EXIT
  problems = 0
  for revision in revisions(args[0], args[1]):
    for problem in validate(_git("log", "-1", "--format=%B", revision)):
      print(f"{revision[:12]} {problem}")
      problems += 1
  return 1 if problems else 0


if __name__ == "__main__":
  raise SystemExit(main())
