"""Tests for the staged-data LGPD guard (tests folder)."""

from check_staged_data import main


def test_no_paths_passes() -> None:
  assert main([]) == 0


def test_data_path_fails() -> None:
  assert main(["data/raw/datajud/2026-11-24/p1.json"]) == 1


def test_multiple_paths_fail() -> None:
  assert main(["data/processed/cases/x.json", "data/manifest.json"]) == 1
