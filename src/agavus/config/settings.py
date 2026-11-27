"""Typed application settings loaded from the environment."""

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_NAME = "world"
NAME_ENV_VAR = "AGAVUS_NAME"
DATA_DIR_ENV_VAR = "AGAVUS_DATA_DIR"
DATA_START_ENV_VAR = "AGAVUS_DATA_START"
DATAJUD_KEY_ENV_VAR = "DATAJUD_API_KEY"
MARITACA_KEY_ENV_VAR = "MARITACA_API_KEY"
DEFAULT_DATA_DIR = "data"
DEFAULT_DATA_START = "2020-01-01"
# CNJ's published public key for the DataJud public API — not a secret.
DEFAULT_DATAJUD_KEY = "cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw=="


@dataclass(frozen=True)
class Settings:
  """Immutable, fully typed application configuration."""

  name: str
  data_dir: Path
  data_start: date
  datajud_api_key: str
  learned_device: str = "auto"
  learned_budget_minutes: float = 120.0
  # Lab teacher for Agathe's chat tuning (docs/llm-lab.md §9.5); never shown in repr.
  maritaca_api_key: str = field(default="", repr=False)


def load_settings() -> Settings:
  """Load settings from `.env` (if present) and the process environment."""
  load_dotenv()
  name = os.environ.get(NAME_ENV_VAR, DEFAULT_NAME).strip()
  data_dir = os.environ.get(DATA_DIR_ENV_VAR, DEFAULT_DATA_DIR).strip()
  start = os.environ.get(DATA_START_ENV_VAR, DEFAULT_DATA_START).strip()
  key = os.environ.get(DATAJUD_KEY_ENV_VAR, DEFAULT_DATAJUD_KEY).strip()
  return Settings(
    name=name or DEFAULT_NAME,
    data_dir=Path(data_dir or DEFAULT_DATA_DIR),
    data_start=date.fromisoformat(start or DEFAULT_DATA_START),
    datajud_api_key=key or DEFAULT_DATAJUD_KEY,
    learned_device=os.environ.get("AGAVUS_LEARNED_DEVICE", "auto").strip() or "auto",
    learned_budget_minutes=float(os.environ.get("AGAVUS_LEARNED_BUDGET_MINUTES", "120")),
    maritaca_api_key=os.environ.get(MARITACA_KEY_ENV_VAR, "").strip(),
  )
