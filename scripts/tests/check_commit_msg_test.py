"""Tests for the commit-message convention checker (tests folder)."""

from pathlib import Path

from check_commit_msg import MAX_HEADER, main, validate


def test_valid_minimal_header() -> None:
  assert validate("feat(corpus): add DataJud ingestion\n") == []


def test_valid_with_body_and_comments() -> None:
  message = "design(decide): narrow scope to F+M\n\n# comment line\nWhy: overlap.\n"
  assert validate(message) == []


def test_passthrough_prefixes() -> None:
  assert validate('Merge branch "main" into feature\n') == []
  assert validate('Revert "feat(corpus): add ingestion"\n') == []


def test_empty_message_fails() -> None:
  assert validate("# only comments\n") == ["empty commit message"]


def test_bad_header_shape_fails() -> None:
  problems = validate("updated stuff\n")
  assert len(problems) == 1
  assert "type(scope): subject" in problems[0]


def test_unknown_type_and_scope_fail() -> None:
  problems = validate("feat(unknown): add thing\n")
  assert any("unknown scope" in problem for problem in problems)
  problems = validate("yolo(corpus): add thing\n")
  assert any("unknown type" in problem for problem in problems)


def test_long_header_fails() -> None:
  header = "feat(corpus): " + "x" * MAX_HEADER
  assert any("max" in problem for problem in validate(header + "\n"))


def test_trailing_period_fails() -> None:
  assert validate("feat(corpus): add thing.\n") == ["subject must not end with a period"]


def test_missing_blank_line_fails() -> None:
  problems = validate("feat(corpus): add thing\nbody right away\n")
  assert problems == ["blank line required between header and body"]


def test_long_body_line_fails() -> None:
  message = "feat(corpus): add thing\n\n" + "y" * 101 + "\n"
  assert any("body line" in problem for problem in validate(message))


def test_main_reads_file(tmp_path: Path) -> None:
  good = tmp_path / "good"
  good.write_text("feat(corpus): add thing\n", encoding="utf-8")
  assert main([str(good)]) == 0
  bad = tmp_path / "bad"
  bad.write_text("nope\n", encoding="utf-8")
  assert main([str(bad)]) == 1
