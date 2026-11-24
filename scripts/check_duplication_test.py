"""Tests for the duplication detector (co-located with the module)."""

from pathlib import Path

from check_duplication import MIN_TOKENS, find_clones, main
from pyutils import code_tokens


def _long_body(offset: int) -> str:
  lines = [f"  value_{i} = {i + offset}" for i in range(MIN_TOKENS)]
  return "def f() -> None:\n" + "\n".join(lines) + "\n"


def test_identical_files_are_clones() -> None:
  source = _long_body(0)
  assert find_clones([code_tokens(source), code_tokens(source)])


def test_distinct_files_pass() -> None:
  assert find_clones([code_tokens(_long_body(0)), code_tokens(_long_body(1000))]) == []


def test_comments_are_ignored() -> None:
  plain = _long_body(0)
  commented = plain.replace("\n", "\n# comment\n")
  assert find_clones([code_tokens(plain), code_tokens(commented)])


def test_duplication_within_one_file() -> None:
  source = _long_body(0) + _long_body(0)
  assert find_clones([code_tokens(source)])


def test_short_shared_runs_pass() -> None:
  small = "def f() -> int:\n  return 1\n"
  assert find_clones([code_tokens(small), code_tokens(small)]) == []


def test_main_exit_codes(tmp_path: Path) -> None:
  (tmp_path / "a.py").write_text(_long_body(0), encoding="utf-8")
  (tmp_path / "b.py").write_text(_long_body(0), encoding="utf-8")
  assert main([str(tmp_path)]) == 1
  (tmp_path / "b.py").write_text(_long_body(1000), encoding="utf-8")
  assert main([str(tmp_path)]) == 0
