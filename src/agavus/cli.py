"""CLI: greeting by default; corpus, model, and answer subcommands.

Global flags: --explain (decision trace) and --json (typed output),
per docs/cli.md. Exit codes for -p: 0 accept · 10 defer · 20 reject · 1 error.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from agavus.answer.compose import compose
from agavus.classify.pipeline import TrainedClassifier, train_classifier
from agavus.classify.seed import SEED_QUERIES
from agavus.classify.store import load_model, save_model
from agavus.config.settings import Settings, load_settings
from agavus.corpus.fetch import FetchError
from agavus.corpus.ingest import import_statute_file, ingest_datajud, ingest_statutes
from agavus.corpus.storage import CorpusStorage
from agavus.decide.rule import decide
from agavus.evals.goldset import GOLD_VERSION, ROUTING_V1
from agavus.evals.run import evaluate, gold_mappings
from agavus.greeting.greet import greet
from agavus.publish.export import stage_release, upload
from agavus.retrieve.search import search
from agavus.taxonomy.labeling import LabelDecision, label_case, label_text

DEFAULT_LIMIT = 500


def _print_json(payload: dict[str, object]) -> None:
  print(json.dumps(payload, ensure_ascii=False, indent=2))


def _probabilities(classifier: TrainedClassifier, text: str) -> dict[str, float]:
  probs = classifier.predict_proba([text])[0]
  return {label: float(value) for label, value in zip(classifier.model.classes, probs, strict=True)}


def _trace(
  classifier: TrainedClassifier, text: str, mapping: dict[str, float]
) -> dict[str, object]:
  return {
    "probabilities": mapping,
    "expected_losses": decide(mapping).expected_losses,
    "top_features": [
      {"term": term, "contribution": value} for term, value in classifier.top_features(text)
    ],
  }


def _print_trace(classifier: TrainedClassifier, text: str, mapping: dict[str, float]) -> None:
  trace = _trace(classifier, text, mapping)
  losses = trace["expected_losses"]
  features = trace["top_features"]
  print("-- explain --")
  print("probabilities: " + " · ".join(f"{k} {v:.3f}" for k, v in mapping.items()))
  if isinstance(losses, dict):
    print("expected losses: " + " · ".join(f"{k} {v:.2f}" for k, v in losses.items()))
  if isinstance(features, list):
    rendered = ", ".join(f"{f['term']} ({f['contribution']:+.2f})" for f in features)
    print(f"top features: {rendered}")


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


def _load_classifier(settings: Settings, command: str) -> TrainedClassifier | None:
  db_path = settings.data_dir / "models" / "models.db"
  if not db_path.exists():
    print(f"{command}: no trained model — run `agavus train` first")
    return None
  return load_model(db_path, "classifier")


def _run_classify(settings: Settings, text: str, explain: bool, as_json: bool) -> int:
  classifier = _load_classifier(settings, "classify")
  if classifier is None:
    return 1
  mapping = _probabilities(classifier, text)
  decision = decide(mapping)
  if as_json:
    payload: dict[str, object] = {"probabilities": mapping, "decision": decision.action}
    if explain:
      payload["trace"] = _trace(classifier, text, mapping)
    _print_json(payload)
    return 0
  rendered = " · ".join(f"{label} {prob:.3f}" for label, prob in mapping.items())
  print(f"classify: {rendered}")
  print(f"decision: {decision.action}")
  if explain:
    _print_trace(classifier, text, mapping)
  return 0


def _run_eval(settings: Settings, as_json: bool) -> int:
  classifier = _load_classifier(settings, "eval")
  if classifier is None:
    return 1
  report = evaluate(gold_mappings(classifier), ROUTING_V1)
  if as_json:
    _print_json({"gold_version": GOLD_VERSION, **asdict(report)})
    return 0
  print(
    f"eval ({GOLD_VERSION}, {len(ROUTING_V1)} items): "
    f"acc {report.accuracy:.2f} · F1 {report.macro_f1:.2f} · M-recall {report.m_recall:.2f}\n"
    f"decisions: {report.accepted} accepted ({report.accepted_correct} correct) · "
    f"{report.deferred} deferred · {report.rejected} rejected\n"
    f"precision-on-accepted {report.precision_on_accepted:.2f} · "
    f"coverage {report.coverage:.2f}"
  )
  return 0


def _run_search(settings: Settings, query: str, limit: int, as_json: bool) -> int:
  storage = CorpusStorage(settings.data_dir)
  hits = search(storage, query, limit=limit)
  if as_json:
    _print_json({"hits": [asdict(hit) for hit in hits]})
    return 0
  if not hits:
    print("search: no hits — ingest + label + rebuild the corpus first")
    return 0
  for hit in hits:
    print(f"{hit.score:6.2f}  {hit.origin_ref}")
    print(f"        {hit.snippet[:120]}…")
  return 0


def _run_ask(settings: Settings, question: str, explain: bool, as_json: bool) -> int:
  classifier = _load_classifier(settings, "ask")
  if classifier is None:
    return 1
  mapping = _probabilities(classifier, question)
  answer = compose(question, mapping, CorpusStorage(settings.data_dir))
  accepted = answer.decision.startswith("accept-")
  if as_json:
    payload: dict[str, object] = {
      "decision": answer.decision,
      "area": answer.decision.removeprefix("accept-") if accepted else None,
      "confidence": max(mapping.values()),
      "text": answer.text,
      "sources": [{"ref": ref} for ref in answer.sources],
    }
    if explain:
      payload["trace"] = _trace(classifier, question, mapping)
    _print_json(payload)
  else:
    print(answer.text)
    if explain:
      _print_trace(classifier, question, mapping)
  if accepted:
    return 0
  return 10 if answer.decision == "defer" else 20


def _run_publish(settings: Settings, repo: str | None) -> int:
  staging = settings.data_dir / "hf-staging"
  try:
    report = stage_release(settings, staging)
  except (KeyError, ValueError) as exc:
    print(f"publish: blocked — {exc}")
    return 1
  print(
    f"publish: staged at {staging} · precision {report.precision_on_accepted:.2f} · "
    f"M-recall {report.m_recall:.2f} · coverage {report.coverage:.2f}"
  )
  if repo is None:
    print("publish: dry run (pass --repo OWNER/NAME to upload; requires `hf auth login`)")
    return 0
  try:
    url = upload(staging, repo)
  except Exception as exc:
    print(f"publish: upload failed — {exc}")
    return 1
  print(f"publish: live at {url}")
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  """Entry point: no args greets; subcommands run corpus pipelines."""
  parser = argparse.ArgumentParser(prog="agavus", description="agavus legal engine")
  parser.add_argument("-p", "--pergunta", default=None, help="ask a legal question")
  parser.add_argument("--explain", action="store_true", help="show the decision trace")
  parser.add_argument("--json", action="store_true", dest="as_json", help="typed JSON output")
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument("--explain", action="store_true", default=argparse.SUPPRESS)
  common.add_argument("--json", action="store_true", dest="as_json", default=argparse.SUPPRESS)
  sub = parser.add_subparsers(dest="command")
  ingest = sub.add_parser(
    "ingest", parents=[common], help="ingest corpus sources into the data dir"
  )
  ingest.add_argument("--source", choices=("datajud", "statutes"), required=True)
  ingest.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
  ingest.add_argument("--raw-file", default=None, help="offline statute payload path")
  ingest.add_argument("--statute-name", default=None, help="e.g. 'Lei 12.318/2010'")
  sub.add_parser("label", parents=[common], help="label stored records (TPU + organ rules)")
  sub.add_parser("train", parents=[common], help="train the classifier on labeled texts")
  classify = sub.add_parser(
    "classify", parents=[common], help="classify a text with the trained model"
  )
  classify.add_argument("text")
  sub.add_parser("eval", parents=[common], help="evaluate the trained model on the gold set")
  search_cmd = sub.add_parser(
    "search", parents=[common], help="BM25 + citation search over the corpus"
  )
  search_cmd.add_argument("query")
  search_cmd.add_argument("--limit", type=int, default=5)
  publish = sub.add_parser(
    "publish", parents=[common], help="stage (and optionally upload) the HF release"
  )
  publish.add_argument(
    "--repo", default=None, help="HF model repo, e.g. Falconiere/agavus-classifier"
  )
  args = parser.parse_args(argv)
  settings = load_settings()
  if args.pergunta is not None:
    return _run_ask(settings, args.pergunta, args.explain, args.as_json)
  handlers = {
    "ingest": lambda: _run_ingest(settings, args.source, args.limit, args),
    "label": lambda: _run_label(settings),
    "train": lambda: _run_train(settings),
    "classify": lambda: _run_classify(settings, args.text, args.explain, args.as_json),
    "eval": lambda: _run_eval(settings, args.as_json),
    "search": lambda: _run_search(settings, args.query, args.limit, args.as_json),
    "publish": lambda: _run_publish(settings, args.repo),
  }
  handler = handlers.get(args.command)
  if handler is not None:
    return handler()
  print(greet(settings.name))
  return 0
