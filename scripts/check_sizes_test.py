"""Tests for the size-limit checker (co-located with the module)."""

from pathlib import Path

from check_sizes import MAX_FILE_LINES, MAX_FUNCTION_LINES, find_violations


def _function_source(body_lines: int) -> str:
  body = "\n".join(f"  value_{i} = {i}" for i in range(body_lines))
  return f"def big() -> None:\n{body}\n"


def test_small_function_passes() -> None:
  assert find_violations(Path("ok.py"), "def f() -> int:\n  return 1\n") == []


def test_long_function_fails() -> None:
  source = _function_source(MAX_FUNCTION_LINES)
  violations = find_violations(Path("big.py"), source)
  assert len(violations) == 1
  assert "function 'big'" in violations[0]


def test_comments_do_not_count() -> None:
  comments = "\n".join(f"  # comment {i}" for i in range(MAX_FUNCTION_LINES))
  source = f"def f() -> int:\n{comments}\n  return 1\n"
  assert find_violations(Path("ok.py"), source) == []


def test_blank_lines_do_not_count() -> None:
  body = "\n\n".join(f"  value_{i} = {i}" for i in range(MAX_FUNCTION_LINES - 1))
  source = f"def big() -> None:\n{body}\n"
  assert find_violations(Path("ok.py"), source) == []


def test_long_file_fails() -> None:
  source = "\n".join(f"value_{i} = {i}" for i in range(MAX_FILE_LINES + 1))
  violations = find_violations(Path("huge.py"), source)
  assert len(violations) == 1
  assert "file has" in violations[0]
