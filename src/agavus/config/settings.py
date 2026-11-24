"""Typed application settings loaded from the environment."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_NAME = "world"
NAME_ENV_VAR = "AGAVUS_NAME"


@dataclass(frozen=True)
class Settings:
  """Immutable, fully typed application configuration."""

  name: str


def load_settings() -> Settings:
  """Load settings from `.env` (if present) and the process environment."""
  load_dotenv()
  raw = os.environ.get(NAME_ENV_VAR, DEFAULT_NAME).strip()
  return Settings(name=raw or DEFAULT_NAME)
