"""Tests for environment-driven settings (co-located with the module)."""

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
