"""Tests for the pre-upload privacy scan (synthetic data only)."""

from pathlib import Path

import pytest

from agavus.publish.privacy import ensure_no_personal_data, scan_tree


def test_scan_flags_personal_data(tmp_path: Path) -> None:
  (tmp_path / "README.md").write_text("contato fulano@example.com", encoding="utf-8")
  assert scan_tree(tmp_path) == [tmp_path / "README.md"]
  with pytest.raises(ValueError, match="personal data"):
    ensure_no_personal_data(tmp_path)


def test_scan_ignores_url_paths_and_test_folders(tmp_path: Path) -> None:
  (tmp_path / "notes.md").write_text(
    "Ver ../../_Ato2019-2022/2021/Lei/L14195.htm e https://example.com/2019-2022",
    encoding="utf-8",
  )
  tests = tmp_path / "tests"
  tests.mkdir()
  (tests / "fixture.py").write_text("contato fulano@example.com", encoding="utf-8")
  assert scan_tree(tmp_path) == []


def test_scan_reads_html_as_prose(tmp_path: Path) -> None:
  asset = tmp_path / "law.html"
  asset.write_text(
    '<a href="../../_Ato2019-2022/2021/Lei/L14195.htm">Art. 1º Esta lei dispõe.</a>',
    encoding="utf-8",
  )
  assert scan_tree(tmp_path) == []


def test_scan_skips_binary_artifacts(tmp_path: Path) -> None:
  (tmp_path / "models.db").write_bytes(b"\x00 CPF 123.456.789-09")
  assert scan_tree(tmp_path) == []


def test_scan_still_flags_data_next_to_slashes(tmp_path: Path) -> None:
  report = tmp_path / "report.md"
  report.write_text("CPF 123.456.789-09/2024", encoding="utf-8")
  assert scan_tree(tmp_path) == [report]
