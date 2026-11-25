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
from agavus.classify.pipeline import TrainedClassifier, train_classifier, training_set
from agavus.classify.store import load_model, save_model
from agavus.config.settings import Settings, load_settings
from agavus.corpus.batch import IngestBatch
from agavus.corpus.djen import ingest_djen
from agavus.corpus.fetch import FetchError
from agavus.corpus.ingest import import_statute_file, ingest_datajud, ingest_statutes
from agavus.corpus.storage import CorpusStorage
from agavus.decide.rule import decide
from agavus.evals.goldset import GOLD_VERSION, ROUTING_V1
from agavus.evals.retrieval import TOP_K, evaluate_retrieval
from agavus.evals.retrievalset import DEV, RETRIEVAL_VERSION
from agavus.evals.run import evaluate, gold_mappings
from agavus.greeting.greet import greet
from agavus.publish.bootstrap import bootstrap_corpus
from agavus.publish.export import stage_release, upload
from agavus.publish.space_release import stage_space, upload_space
from agavus.retrieve.dense import index_statutes, search_statutes
from agavus.retrieve.encoder import download_encoder, load_encoder
from agavus.retrieve.search import search

DEFAULT_LIMIT = 500
RETRIEVAL_DEPTH = 10
SPACE_DIR = Path("space")


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


def _print_batch(source: str, batch: IngestBatch) -> None:
  status = "done" if batch.done else "partial"
  print(f"ingest {source}: {batch.new} new records; {status}; cursor {batch.cursor}")
  for line in batch.unknown:
    print(f"unknown tpu: {line}")


def _run_ingest(settings: Settings, source: str, limit: int, args: argparse.Namespace) -> int:
  if args.pages is not None and args.pages < 1:
    print("--pages must be >= 1")
    return 1
  if source == "statutes" and args.raw_file is not None:
    if not args.statute_name:
      print("--raw-file requires --statute-name")
      return 1
    new = import_statute_file(settings, args.statute_name, Path(args.raw_file))
    print(f"import {args.statute_name}: {new} new records")
    return 0
  try:
    if source == "datajud":
      batch = ingest_datajud(settings, limit, pages=args.pages)
    elif source == "djen":
      batch = ingest_djen(settings, pages=args.pages)
    else:
      new = ingest_statutes(settings)
      print(f"ingest statutes: {new} new records")
      return 0
  except FetchError as exc:
    print(f"ingest {source} failed (source unavailable, nothing saved): {exc}")
    return 1
  _print_batch(source, batch)
  return 0


def _run_label(settings: Settings) -> int:
  storage = CorpusStorage(settings.data_dir)
  decisions = storage.run_labeling()
  review = sum(decision.needs_review for decision in decisions)
  print(f"label: {len(decisions)} records, {review} to review")
  return 0


def _run_train(settings: Settings) -> int:
  storage = CorpusStorage(settings.data_dir)
  texts, targets = training_set(storage)
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


def _run_embed(settings: Settings) -> int:
  download_encoder(settings.data_dir)
  encoder = load_encoder(settings.data_dir)
  if encoder is None:
    print("embed: encoder weights not found after download")
    return 1
  count = index_statutes(CorpusStorage(settings.data_dir), encoder)
  print(f"embed: {count} statute articles indexed")
  return 0


def _run_retrieval_eval(settings: Settings) -> int:
  storage = CorpusStorage(settings.data_dir)
  encoder = load_encoder(settings.data_dir)
  report = evaluate_retrieval(
    DEV,
    lambda question: [
      hit.origin_ref for hit in search_statutes(storage, question, RETRIEVAL_DEPTH, encoder)
    ],
  )
  mode = "meaning" if encoder is not None else "words"
  print(
    f"retrieval ({RETRIEVAL_VERSION} dev, {len(DEV)} items, by {mode}): "
    f"first {report.first} · top-{TOP_K} {report.in_top_k} · MRR {report.mrr:.3f}"
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
  storage = CorpusStorage(settings.data_dir)
  answer = compose(question, mapping, storage, load_encoder(settings.data_dir))
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


def _run_publish(
  settings: Settings,
  repo: str | None,
  space_repo: str | None,
  revision: str,
  bootstrap: Path | None,
) -> int:
  if bootstrap is not None:
    try:
      bootstrap_corpus(settings, bootstrap)
    except (KeyError, OSError, ValueError) as exc:
      print(f"publish: bootstrap failed — {exc}")
      return 1
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
  if repo is not None:
    try:
      url = upload(staging, repo)
    except Exception as exc:
      print(f"publish: upload failed — {exc}")
      return 1
    print(f"publish: model live at {url}")
  if space_repo is not None:
    try:
      space_staging = stage_space(SPACE_DIR, settings.data_dir / "hf-space-staging", revision)
      space_url = upload_space(space_staging, space_repo)
    except Exception as exc:
      print(f"publish: Space upload failed — {exc}")
      return 1
    print(f"publish: Space live at {space_url}")
  if repo is None and space_repo is None:
    print(
      "publish: dry run (pass --repo and/or --space OWNER/NAME to upload; requires `hf auth login`)"
    )
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
  ingest.add_argument("--source", choices=("datajud", "statutes", "djen"), required=True)
  ingest.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
  ingest.add_argument(
    "--pages", type=int, default=None, help="cap pages this run; default walks until empty"
  )
  ingest.add_argument("--raw-file", default=None, help="offline statute payload path")
  ingest.add_argument("--statute-name", default=None, help="e.g. 'Lei 12.318/2010'")
  sub.add_parser("label", parents=[common], help="label stored records (TPU + organ rules)")
  sub.add_parser("train", parents=[common], help="train the classifier on labeled texts")
  classify = sub.add_parser(
    "classify", parents=[common], help="classify a text with the trained model"
  )
  classify.add_argument("text")
  evaluation = sub.add_parser(
    "eval", parents=[common], help="evaluate the trained model on the gold set"
  )
  evaluation.add_argument(
    "--retrieval", action="store_true", help="evaluate article retrieval (dev questions)"
  )
  sub.add_parser("embed", parents=[common], help="download the encoder and index statutes")
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
  publish.add_argument(
    "--space", dest="space_repo", default=None, help="HF Space repo, e.g. Falconiere/agavus"
  )
  publish.add_argument(
    "--revision", default="main", help="git ref the Space installs agavus from (tag/sha)"
  )
  publish.add_argument(
    "--bootstrap", type=Path, default=None, help="assets dir for an offline public-corpus build"
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
    "eval": lambda: (
      _run_retrieval_eval(settings) if args.retrieval else _run_eval(settings, args.as_json)
    ),
    "embed": lambda: _run_embed(settings),
    "search": lambda: _run_search(settings, args.query, args.limit, args.as_json),
    "publish": lambda: _run_publish(
      settings, args.repo, args.space_repo, args.revision, args.bootstrap
    ),
  }
  handler = handlers.get(args.command)
  if handler is not None:
    return handler()
  print(greet(settings.name))
  return 0
