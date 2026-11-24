"""CLI: greeting by default; `ingest` and `label` subcommands."""

import argparse
from collections.abc import Sequence

from agavus.config.settings import Settings, load_settings
from agavus.corpus.fetch import FetchError
from agavus.corpus.ingest import ingest_datajud, ingest_statutes
from agavus.corpus.storage import CorpusStorage
from agavus.greeting.greet import greet
from agavus.taxonomy.labeling import LabelDecision, label_case, label_text

DEFAULT_LIMIT = 500


def _run_ingest(settings: Settings, source: str, limit: int) -> int:
  try:
    new = ingest_datajud(settings, limit) if source == "datajud" else ingest_statutes(settings)
  except FetchError as exc:
    print(f"ingest {source} failed (source unavailable, nothing saved): {exc}")
    return 1
  print(f"ingest {source}: {new} new records")
  return 0


def _run_label(settings: Settings) -> int:
  storage = CorpusStorage(settings.data_dir)
  decisions: list[LabelDecision] = [label_case(record) for record in storage.load_cases()]
  decisions += [label_text(document) for document in storage.load_texts()]
  index = {
    decision.record_id: {
      "label": decision.domain_label,
      "source": decision.label_source.value,
      "needs_review": decision.needs_review,
      "reason": decision.reason,
    }
    for decision in decisions
  }
  storage.save_index("labels.json", index)
  review = sum(decision.needs_review for decision in decisions)
  print(f"label: {len(decisions)} records, {review} to review")
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  """Entry point: no args greets; subcommands run corpus pipelines."""
  parser = argparse.ArgumentParser(prog="agavus", description="agavus legal engine")
  sub = parser.add_subparsers(dest="command")
  ingest = sub.add_parser("ingest", help="ingest corpus sources into the data dir")
  ingest.add_argument("--source", choices=("datajud", "statutes"), required=True)
  ingest.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
  sub.add_parser("label", help="label stored records (TPU + organ rules)")
  args = parser.parse_args(argv)
  settings = load_settings()
  if args.command == "ingest":
    return _run_ingest(settings, args.source, args.limit)
  if args.command == "label":
    return _run_label(settings)
  print(greet(settings.name))
  return 0
