"""Isolated command; never imported by application startup."""

import argparse
from dataclasses import replace

from agavus.classify.learned.runner import experiment
from agavus.config.settings import load_settings


def main() -> None:
  """Run one private comparison with explicit local computation limits."""
  settings = load_settings()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=("experiment",))
  parser.add_argument("--device", choices=("auto", "cpu", "mps"), default=settings.learned_device)
  parser.add_argument("--budget-minutes", type=float, default=settings.learned_budget_minutes)
  args = parser.parse_args()
  folder = experiment(
    replace(settings, learned_device=args.device, learned_budget_minutes=args.budget_minutes)
  )
  print(f"Private experiment: {folder / 'report.md'}")


if __name__ == "__main__":
  main()
