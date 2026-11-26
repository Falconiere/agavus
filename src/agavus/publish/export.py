"""Eval-gated release staging and upload (docs/publishing.md).

No publish without a green eval report. Staging lives outside git;
upload requires explicit repo + authentication.
"""

import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import HfApi

from agavus.classify.pipeline import TrainReport
from agavus.classify.store import load_model
from agavus.config.settings import Settings
from agavus.corpus.storage import CorpusStorage
from agavus.evals.goldset import GOLD_VERSION, ROUTING_V1
from agavus.evals.run import EvalReport, evaluate, gold_mappings
from agavus.publish.card import build_model_card
from agavus.publish.privacy import ensure_no_personal_data

MIN_M_RECALL = 0.98
MIN_PRECISION = 1.0
REPORT_DECIMALS = 6


def gate(report: EvalReport) -> list[str]:
  """Hard release gates; empty list means publishable."""
  problems: list[str] = []
  if report.m_recall < MIN_M_RECALL:
    problems.append(f"M-recall {report.m_recall:.2f} < {MIN_M_RECALL}")
  if report.precision_on_accepted < MIN_PRECISION:
    problems.append(f"precision-on-accepted {report.precision_on_accepted:.2f} < 1.00")
  if report.accepted == 0:
    problems.append("nothing accepted — artifact has no demonstrated coverage")
  return problems


def corpus_stats(storage: CorpusStorage) -> dict[str, int]:
  """Document counts by source and label area (provenance for the card)."""
  stats: dict[str, int] = {}
  for document in storage.load_texts():
    key = f"source:{document.source.value}"
    stats[key] = stats.get(key, 0) + 1
  for decision in storage.load_labels():
    if decision.domain_label is not None:
      key = f"area:{decision.domain_label}"
      stats[key] = stats.get(key, 0) + 1
  return stats


def _latest_report(db_path: Path) -> TrainReport:
  with closing(sqlite3.connect(db_path)) as conn:
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM train_reports ORDER BY id DESC LIMIT 1").fetchone()
  if row is None:
    raise KeyError("no training report found — run `agavus train` first")
  return TrainReport(
    holdout_accuracy=float(row["holdout_accuracy"]),
    holdout_macro_f1=float(row["holdout_macro_f1"]),
    holdout_ece=float(row["holdout_ece"]),
    temperature=float(row["temperature"]),
    n_train=int(row["n_train"]),
    n_holdout=int(row["n_holdout"]),
    n_calibration=int(row["n_calibration"]) if "n_calibration" in set(row.keys()) else 0,
  )


def _report_json(report: EvalReport) -> str:
  """Typed eval report JSON; 6 decimals is plenty and keeps artifacts clean."""
  payload = {
    key: round(value, REPORT_DECIMALS) if isinstance(value, float) else value
    for key, value in asdict(report).items()
  }
  return json.dumps(payload, ensure_ascii=False, indent=2)


def stage_release(settings: Settings, staging: Path) -> EvalReport:
  """Build the release folder; raises on gate failure (nothing staged)."""
  db_path = settings.data_dir / "models" / "models.db"
  classifier = load_model(db_path, "classifier")
  report = evaluate(gold_mappings(classifier), ROUTING_V1)
  problems = gate(report)
  if problems:
    raise ValueError("release gate failed: " + "; ".join(problems))
  staging.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(db_path, staging / "models.db")
  card = build_model_card(
    train_report=_latest_report(db_path),
    eval_report=report,
    corpus_stats=corpus_stats(CorpusStorage(settings.data_dir)),
    gold_version=GOLD_VERSION,
  )
  (staging / "README.md").write_text(card, encoding="utf-8")
  (staging / "eval_report.json").write_text(_report_json(report), encoding="utf-8")
  ensure_no_personal_data(staging)
  return report


def upload(staging: Path, repo: str) -> str:
  """Create/update the HF model repo and upload the staged release."""
  ensure_no_personal_data(staging)
  api = HfApi()
  api.create_repo(repo, repo_type="model", exist_ok=True)
  api.upload_folder(repo_id=repo, folder_path=str(staging), repo_type="model")
  return f"https://huggingface.co/{repo}"
