"""Enforce size limits: functions <= 80 code lines, files <= 500 code lines.

Blank lines and comments are excluded from the counts.
"""

import argparse
import ast
from collections.abc import Sequence
from pathlib import Path

from pyutils import code_lines, iter_python_files

MAX_FUNCTION_LINES = 80
MAX_FILE_LINES = 500
DEFAULT_ROOTS = ("src", "scripts")


def find_violations(path: Path, source: str) -> list[str]:
  """Return all size-limit violations for one Python source file."""
  lines = code_lines(source)
  violations: list[str] = []
  if len(lines) > MAX_FILE_LINES:
    violations.append(f"{path}: file has {len(lines)} code lines (max {MAX_FILE_LINES})")
  for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
      end = node.end_lineno or node.lineno
      count = sum(1 for n in range(node.lineno, end + 1) if n in lines)
      if count > MAX_FUNCTION_LINES:
        violations.append(
          f"{path}:{node.lineno}: function '{node.name}' has {count} code lines "
          f"(max {MAX_FUNCTION_LINES})"
        )
  return violations


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point; returns 1 when any violation is found."""
  parser = argparse.ArgumentParser(description="Check function/file size limits.")
  parser.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
  args = parser.parse_args(argv)
  violations: list[str] = []
  for path in iter_python_files(args.roots):
    violations.extend(find_violations(path, path.read_text(encoding="utf-8")))
  for violation in violations:
    print(violation)
  return 1 if violations else 0


if __name__ == "__main__":
  raise SystemExit(main())
