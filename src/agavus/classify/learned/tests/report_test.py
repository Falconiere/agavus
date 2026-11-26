"""Safety verdict and report-persistence regression tests."""

from pathlib import Path

from agavus.classify.learned.report import verdict, write_report
from agavus.classify.learned.types import SEEDS

M_RECALL_TARGET = 0.98


def _metrics(*, unsafe: list[str] | None = None, recall: float = 0.99) -> dict[str, object]:
  return {
    "macro_f1": 0.9,
    "m_recall": recall,
    "unsafe_m_decisions": unsafe or [],
    "ece": 0.01,
    "accepted_precision": 0.99,
    "coverage": 0.5,
    "m_recall_target_met": recall >= M_RECALL_TARGET,
  }


def _results(
  *, unsafe: list[str] | None = None, recall: float = 0.99
) -> dict[str, dict[str, object]]:
  result: dict[str, dict[str, object]] = {
    "tfidf": {"area": _metrics() | {"macro_f1": 0.7}, "relation": _metrics()},
    "frozen": {"area": _metrics() | {"macro_f1": 0.8}, "relation": _metrics()},
  }
  for seed in SEEDS:
    result[f"area_only:{seed}"] = {
      "area": _metrics(unsafe=unsafe, recall=recall),
      "relation": _metrics(),
    }
    result[f"joint:{seed}"] = {
      "area": _metrics(unsafe=unsafe, recall=recall),
      "relation": _metrics(),
    }
  return result


def test_verdict_never_declares_partial_or_unsafe_or_weak_recall_winner(tmp_path: Path) -> None:
  assert verdict(_results(), complete=True)["winner"] == "area_only"
  assert verdict(_results(), complete=False)["winner"] is None
  assert verdict(_results(unsafe=["case"]), complete=True)["winner"] is None
  assert verdict(_results(recall=0.97), complete=True)["winner"] is None
  report = tmp_path / "report"
  report.mkdir()
  measured = _results()
  measured["tfidf"]["cpu_latency"] = {"median_seconds": 0.01, "max_seconds": 0.02}
  measured["tfidf"]["relation"] = _metrics() | {"category_failures": {"quotation": ["pair"]}}
  write_report(report, measured, {"winner": None, "status": "partial"})
  receipt = (report / "report.md").read_text()
  assert "partial" in receipt
  assert "CPU median/max: 0.0100s / 0.0200s" in receipt
  assert "quotation: 1" in receipt
