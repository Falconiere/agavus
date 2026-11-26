"""Conservative comparison verdicts and a small private human-readable receipt."""

import json
from pathlib import Path

import numpy as np

from agavus.classify.learned.metrics import ECE_TARGET, PRECISION_TARGET
from agavus.classify.learned.types import SEEDS
from agavus.corpus.jsonutil import as_dict


def number(mapping: dict[str, object], key: str) -> float:
  """Reject malformed measurements rather than silently accepting a winner."""
  value = mapping[key]
  if not isinstance(value, int | float) or not np.isfinite(value):
    raise ValueError(f"nonfinite or missing numeric measurement: {key}")
  return float(value)


def verdict(results: dict[str, dict[str, object]], *, complete: bool) -> dict[str, object]:
  """A full paired comparison across all three seeds is required for a winner."""
  promising: list[str] = []
  variation: dict[str, object] = {}
  if not complete:
    return {"winner": None, "reason": "partial comparison; no winner", "seed_variability": {}}
  baselines = [as_dict(results[name]["area"]) for name in ("tfidf", "frozen")]
  for variant in ("area_only", "joint"):
    measurements = [as_dict(results[f"{variant}:{seed}"]["area"]) for seed in SEEDS]
    values = [number(m, "macro_f1") for m in measurements]
    variation[variant] = {
      "macro_f1_mean": float(np.mean(values)),
      "macro_f1_std": float(np.std(values)),
      "m_recall_range": [
        min(number(m, "m_recall") for m in measurements),
        max(number(m, "m_recall") for m in measurements),
      ],
      "m_recall_target_met_all_seeds": all(m["m_recall_target_met"] for m in measurements),
    }
    passed = all(
      number(m, "macro_f1") > max(number(b, "macro_f1") for b in baselines)
      and number(m, "m_recall") >= max(number(b, "m_recall") for b in baselines)
      and not m["unsafe_m_decisions"]
      and number(m, "ece") <= ECE_TARGET
      and number(m, "accepted_precision") >= PRECISION_TARGET
      for m in measurements
    )
    if variant == "joint":
      relation_floor = max(
        number(as_dict(results[b]["relation"]), "macro_f1") for b in ("tfidf", "frozen")
      )
      passed = passed and all(
        number(as_dict(results[f"joint:{s}"]["relation"]), "macro_f1") > relation_floor
        for s in SEEDS
      )
    if passed:
      promising.append(variant)
  winner = (
    max(
      promising,
      key=lambda v: float(
        np.mean([number(as_dict(results[f"{v}:{s}"]["area"]), "macro_f1") for s in SEEDS])
      ),
    )
    if promising
    else None
  )
  return {
    "winner": winner,
    "promising": promising,
    "seed_variability": variation,
    "reason": "experimental evidence only; weak labels and synthetic relations",
  }


def write_report(
  folder: Path, results: dict[str, dict[str, object]], summary: dict[str, object]
) -> None:
  """Metrics only here; private predictions and provenance remain in SQLite."""
  lines = [
    "# Private learned-classifier experiment",
    "",
    f"Status: {summary.get('status', 'unknown')}. Winner: {summary.get('winner') or 'none'}.",
    "",
    "| Variant | Document F1 | M recall | Unsafe M | ECE | Precision | Coverage | Relation F1 |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
  ]
  for name, result in results.items():
    area, relation = as_dict(result["area"]), as_dict(result["relation"])
    unsafe = area["unsafe_m_decisions"]
    assert isinstance(unsafe, list)
    values = [
      f"{number(area, key):.4f}"
      for key in ("macro_f1", "m_recall", "ece", "accepted_precision", "coverage")
    ]
    lines.append(
      f"| {name} | {values[0]} | {values[1]} | {len(unsafe)} | {values[2]} | "
      f"{values[3]} | {values[4]} | {number(relation, 'macro_f1'):.4f} |"
    )
  lines.extend(["", "## Diagnostics", ""])
  for name, result in results.items():
    latency = as_dict(result.get("cpu_latency"))
    if latency:
      lines.append(
        f"- {name}: CPU median/max: {number(latency, 'median_seconds'):.4f}s / "
        f"{number(latency, 'max_seconds'):.4f}s per document."
      )
    categories = as_dict(as_dict(result["relation"]).get("category_failures"))
    failures = ", ".join(
      f"{category}: {len(ids)}"
      for category, ids in sorted(categories.items())
      if isinstance(ids, list)
    )
    if failures:
      lines.append(f"  Relation errors by category: {failures}.")
    pairs = as_dict(result.get("counterfactuals"))
    failed = pairs.get("relation_flip_and_area_stable_failures")
    if isinstance(failed, list):
      lines.append(
        f"  Grant/denial flip with stable area: {len(failed)} failures / "
        f"{pairs['families']} unseen families."
      )
  lines.extend(
    ["", "## Run summary", "", "```json", json.dumps(summary, indent=2, ensure_ascii=False), "```"]
  )
  lines.extend(
    [
      "",
      "M recall target: 0.98; ECE target: 0.05; accepted precision target: 0.95.",
      "Area-only relation-head scores are untrained-head diagnostics.",
      "Existing query development scores are diagnostic, never fresh gold.",
      "No production activation or uploads. This is not evidence of production safety.",
    ]
  )
  (folder / "report.md").write_text("\n".join(lines) + "\n")
