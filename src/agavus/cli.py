"""CLI: greeting by default; `ingest` and `label` subcommands."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from agavus.classify.pipeline import train_classifier
from agavus.classify.seed import SEED_QUERIES
from agavus.classify.store import load_model, save_model
from agavus.config.settings import Settings, load_settings
from agavus.corpus.fetch import FetchError
from agavus.corpus.ingest import import_statute_file, ingest_datajud, ingest_statutes
from agavus.corpus.storage import CorpusStorage
from agavus.decide.rule import decide
from agavus.evals.goldset import GOLD_VERSION, ROUTING_V1
from agavus.evals.run import evaluate
from agavus.greeting.greet import greet
from agavus.taxonomy.labeling import LabelDecision, label_case, label_text

DEFAULT_LIMIT = 500


def _run_ingest(settings: Settings, source: str, limit: int, args: argparse.Namespace) -> int:
  if source == "statutes" and args.raw_file is not None:
    if not args.statute_name:
      print("--raw-file requires --statute-name")
      return 1
    new = import_statute_file(settings, args.statute_name, Path(args.raw_file))
    print(f"import {args.statute_name}: {new} new records")
    return 0
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
  storage.save_labels(decisions)
  review = sum(decision.needs_review for decision in decisions)
  print(f"label: {len(decisions)} records, {review} to review")
  return 0


def _run_train(settings: Settings) -> int:
  storage = CorpusStorage(settings.data_dir)
  labels = {decision.record_id: decision for decision in storage.load_labels()}
  texts: list[str] = []
  targets: list[str] = []
  for document in storage.load_texts():
    decision = labels.get(document.id)
    if decision is not None and decision.domain_label is not None:
      texts.append(document.text)
      targets.append(decision.domain_label)
  texts += [text for text, _ in SEED_QUERIES]
  targets += [label for _, label in SEED_QUERIES]
  if not texts:
    print("train: no labeled texts — run ingest + label first")
    return 1
  classifier, report = train_classifier(texts, targets)
  db_path = settings.data_dir / "models" / "models.db"
  save_model(db_path, "classifier", classifier, report)
  print(
    f"train: {report.n_train} train / {report.n_holdout} holdout · "
    f"acc {report.holdout_accuracy:.2f} · F1 {report.holdout_macro_f1:.2f} · "
    f"ECE {report.holdout_ece:.3f} · T {report.temperature:.2f} → {db_path}"
  )
  return 0


def _run_classify(settings: Settings, text: str) -> int:
  db_path = settings.data_dir / "models" / "models.db"
  if not db_path.exists():
    print("classify: no trained model — run `agavus train` first")
    return 1
  classifier = load_model(db_path, "classifier")
  probs = classifier.predict_proba([text])[0]
  pairs = sorted(zip(classifier.model.classes, probs, strict=True), key=lambda p: -p[1])
  rendered = " · ".join(f"{label} {prob:.3f}" for label, prob in pairs)
  decision = decide({label: float(prob) for label, prob in pairs})
  print(f"classify: {rendered}")
  print(f"decision: {decision.action}")
  return 0


def _run_eval(settings: Settings) -> int:
  db_path = settings.data_dir / "models" / "models.db"
  if not db_path.exists():
    print("eval: no trained model — run `agavus train` first")
    return 1
  classifier = load_model(db_path, "classifier")
  probs = classifier.predict_proba([item.text for item in ROUTING_V1])
  mappings = [
    {label: float(value) for label, value in zip(classifier.model.classes, row, strict=True)}
    for row in probs
  ]
  report = evaluate(mappings, ROUTING_V1)
  print(
    f"eval ({GOLD_VERSION}, {len(ROUTING_V1)} items): "
    f"acc {report.accuracy:.2f} · F1 {report.macro_f1:.2f} · M-recall {report.m_recall:.2f}\n"
    f"decisions: {report.accepted} accepted ({report.accepted_correct} correct) · "
    f"{report.deferred} deferred · {report.rejected} rejected\n"
    f"precision-on-accepted {report.precision_on_accepted:.2f} · "
    f"coverage {report.coverage:.2f}"
  )
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  """Entry point: no args greets; subcommands run corpus pipelines."""
  parser = argparse.ArgumentParser(prog="agavus", description="agavus legal engine")
  sub = parser.add_subparsers(dest="command")
  ingest = sub.add_parser("ingest", help="ingest corpus sources into the data dir")
  ingest.add_argument("--source", choices=("datajud", "statutes"), required=True)
  ingest.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
  ingest.add_argument("--raw-file", default=None, help="offline statute payload path")
  ingest.add_argument("--statute-name", default=None, help="e.g. 'Lei 12.318/2010'")
  sub.add_parser("label", help="label stored records (TPU + organ rules)")
  sub.add_parser("train", help="train the classifier on labeled texts")
  classify = sub.add_parser("classify", help="classify a text with the trained model")
  classify.add_argument("text")
  sub.add_parser("eval", help="evaluate the trained model on the gold set")
  args = parser.parse_args(argv)
  settings = load_settings()
  if args.command == "ingest":
    return _run_ingest(settings, args.source, args.limit, args)
  if args.command == "label":
    return _run_label(settings)
  if args.command == "train":
    return _run_train(settings)
  if args.command == "classify":
    return _run_classify(settings, args.text)
  if args.command == "eval":
    return _run_eval(settings)
  print(greet(settings.name))
  return 0
