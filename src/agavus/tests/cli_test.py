"""Tests for the CLI (tests folder; ingestion mocked, no network)."""

from pathlib import Path

import pytest

from agavus import cli
from agavus.corpus.fetch import FetchError


def test_default_greets(capsys: pytest.CaptureFixture[str]) -> None:
  assert cli.main([]) == 0
  assert "Hello, world!" in capsys.readouterr().out


def test_ingest_datajud_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
  called: list[int] = []

  def fake(settings: object, limit: int) -> int:
    called.append(limit)
    return 3

  monkeypatch.setattr(cli, "ingest_datajud", fake)
  assert cli.main(["ingest", "--source", "datajud", "--limit", "7"]) == 0
  assert called == [7]


def test_ingest_statutes_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(cli, "ingest_statutes", lambda settings: 5)
  assert cli.main(["ingest", "--source", "statutes"]) == 0


def test_ingest_requires_source() -> None:
  with pytest.raises(SystemExit):
    cli.main(["ingest"])


def test_ingest_fetch_error_is_friendly(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  def failing(settings: object, limit: int) -> int:
    raise FetchError("HTTP 429")

  monkeypatch.setattr(cli, "ingest_datajud", failing)
  assert cli.main(["ingest", "--source", "datajud"]) == 1
  assert "nothing saved" in capsys.readouterr().out


def test_label_empty_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("AGAVUS_DATA_DIR", str(tmp_path))
  assert cli.main(["label"]) == 0
  assert (tmp_path / "processed" / "corpus.db").exists()
