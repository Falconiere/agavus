"""CLI entry point: `python -m agavus` or `agavus`."""

from agavus.config.settings import load_settings
from agavus.greeting.greet import greet


def main() -> None:
  """Print a greeting built from environment-driven settings."""
  settings = load_settings()
  print(greet(settings.name))


if __name__ == "__main__":
  main()
