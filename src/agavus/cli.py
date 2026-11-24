"""CLI: greeting by default; `ingest` subcommand for corpus runs."""

import argparse
from collections.abc import Sequence

from agavus.config.settings import load_settings
from agavus.corpus.fetch import FetchError
from agavus.corpus.ingest import ingest_datajud, ingest_statutes
from agavus.greeting.greet import greet

DEFAULT_LIMIT = 500


def main(argv: Sequence[str] | None = None) -> int:
  """Entry point: no args greets; `ingest --source …` runs the pipeline."""
  parser = argparse.ArgumentParser(prog="agavus", description="agavus legal engine")
  sub = parser.add_subparsers(dest="command")
  ingest = sub.add_parser("ingest", help="ingest corpus sources into the data dir")
  ingest.add_argument("--source", choices=("datajud", "statutes"), required=True)
  ingest.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
  args = parser.parse_args(argv)
  settings = load_settings()
  if args.command == "ingest":
    try:
      if args.source == "datajud":
        new = ingest_datajud(settings, args.limit)
      else:
        new = ingest_statutes(settings)
    except FetchError as exc:
      print(f"ingest {args.source} failed (source unavailable, nothing saved): {exc}")
      return 1
    print(f"ingest {args.source}: {new} new records")
    return 0
  print(greet(settings.name))
  return 0
