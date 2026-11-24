"""Tests for environment-driven settings (in the domain tests folder)."""

from datetime import date
from pathlib import Path

import pytest

from agavus.config import settings


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
  """Isolate tests from any local `.env` file."""
  monkeypatch.setattr(settings, "load_dotenv", lambda: None)


def test_default_name(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv(settings.NAME_ENV_VAR, raising=False)
  assert settings.load_settings().name == "world"


def test_name_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(settings.NAME_ENV_VAR, "Brasil")
  assert settings.load_settings().name == "Brasil"


def test_blank_name_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(settings.NAME_ENV_VAR, "   ")
  assert settings.load_settings().name == "world"


def test_data_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
  for var in (settings.DATA_DIR_ENV_VAR, settings.DATA_START_ENV_VAR):
    monkeypatch.delenv(var, raising=False)
  loaded = settings.load_settings()
  assert loaded.data_dir == Path("data")
  assert loaded.data_start == date(2020, 1, 1)
  assert loaded.datajud_api_key == settings.DEFAULT_DATAJUD_KEY


def test_data_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(settings.DATA_DIR_ENV_VAR, "/tmp/corpus")
  monkeypatch.setenv(settings.DATA_START_ENV_VAR, "2021-06-01")
  monkeypatch.setenv(settings.DATAJUD_KEY_ENV_VAR, "custom-key")
  loaded = settings.load_settings()
  assert loaded.data_dir == Path("/tmp/corpus")
  assert loaded.data_start == date(2021, 6, 1)
  assert loaded.datajud_api_key == "custom-key"
