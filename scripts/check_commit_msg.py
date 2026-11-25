"""Validate commit messages against the repo convention (docs/commits.md).

Commits are project memory: type(scope): subject, body explains why.
"""

import re
import sys
from collections.abc import Sequence
from pathlib import Path

TYPES = ("feat", "fix", "design", "docs", "refactor", "test", "chore", "build")
SCOPES = (
  "corpus",
  "taxonomy",
  "classify",
  "decide",
  "retrieve",
  "reason",
  "answer",
  "publish",
  "evals",
  "config",
  "greeting",
  "lab",
  "docs",
  "scripts",
  "tooling",
  "repo",
)
MAX_HEADER = 72
MAX_BODY_LINE = 100
PASSTHROUGH_PREFIXES = ("Merge ", "Revert ", "fixup! ", "squash! ")
HEADER_RE = re.compile(r"^(?P<type>[a-z]+)\((?P<scope>[a-z0-9-]+)\): (?P<subject>.+)$")


def validate(message: str) -> list[str]:
  """Return convention violations for a commit message (comments stripped)."""
  lines = [line for line in message.splitlines() if not line.startswith("#")]
  text = "\n".join(lines).strip()
  if not text:
    return ["empty commit message"]
  if text.startswith(PASSTHROUGH_PREFIXES):
    return []
  header, *rest = text.split("\n")
  match = HEADER_RE.match(header)
  if match is None:
    return [f"header must be 'type(scope): subject' — got: {header!r}"]
  problems: list[str] = []
  if match["type"] not in TYPES:
    problems.append(f"unknown type '{match['type']}' (allowed: {', '.join(TYPES)})")
  if match["scope"] not in SCOPES:
    problems.append(f"unknown scope '{match['scope']}' (allowed: {', '.join(SCOPES)})")
  if len(header) > MAX_HEADER:
    problems.append(f"header is {len(header)} chars (max {MAX_HEADER})")
  if match["subject"].endswith("."):
    problems.append("subject must not end with a period")
  if rest:
    if rest[0] != "":
      problems.append("blank line required between header and body")
    for number, line in enumerate(rest[1:], start=2):
      if len(line) > MAX_BODY_LINE:
        problems.append(f"body line {number} is {len(line)} chars (max {MAX_BODY_LINE})")
  return problems


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point: validate the commit-msg file passed by pre-commit."""
  args = list(sys.argv[1:] if argv is None else argv)
  if len(args) != 1:
    print("usage: check_commit_msg.py <commit-msg-file>")
    return 2
  problems = validate(Path(args[0]).read_text(encoding="utf-8"))
  for problem in problems:
    print(problem)
  return 1 if problems else 0


if __name__ == "__main__":
  raise SystemExit(main())
