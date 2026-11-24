"""Tests for shared source-analysis helpers (in the domain tests folder)."""

from pyutils import code_lines, code_tokens


def test_comments_and_blanks_excluded_from_lines() -> None:
  source = "# header\n\nx = 1  # trailing\n"
  assert code_lines(source) == {3}


def test_tokens_exclude_comments_and_layout() -> None:
  tokens = code_tokens("# c\nx = 1\n")
  assert [token.string for token in tokens] == ["x", "=", "1"]
