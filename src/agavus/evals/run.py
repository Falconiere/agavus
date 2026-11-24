"""Gold-set evaluation: classification + decision-layer quality.

Pure functions over probabilities and gold items; the CLI wires the trained
model. Gold never trains (firewall, docs/evals.md).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agavus.decide.rule import decide
from agavus.evals.goldset import GoldItem
from agavus.evals.metrics import accuracy, macro_f1


@dataclass(frozen=True)
class EvalReport:
  """Quality summary of one model against one gold set."""

  accuracy: float
  macro_f1: float
  accepted: int
  accepted_correct: int
  deferred: int
  rejected: int
  precision_on_accepted: float
  coverage: float
  m_recall: float


def evaluate(probabilities: Sequence[Mapping[str, float]], gold: Sequence[GoldItem]) -> EvalReport:
  """Argmax classification metrics + three-way decision outcomes."""
  classes = tuple(sorted({item.expected for item in gold}))
  predictions = [
    max(probs, key=lambda state: probs[state]) if probs else "O" for probs in probabilities
  ]
  gold_labels = [item.expected for item in gold]
  decisions = [decide(probs) for probs in probabilities]
  accepted = [
    (pred, item)
    for pred, item, decision in zip(predictions, gold, decisions, strict=True)
    if decision.action.startswith("accept-")
  ]
  accepted_correct = sum(pred == item.expected for pred, item in accepted)
  m_items = [
    (pred, item) for pred, item in zip(predictions, gold, strict=True) if item.expected == "M"
  ]
  m_hits = sum(pred == "M" for pred, _ in m_items)
  total = len(gold)
  return EvalReport(
    accuracy=accuracy(predictions, gold_labels),
    macro_f1=macro_f1(predictions, gold_labels, classes),
    accepted=len(accepted),
    accepted_correct=accepted_correct,
    deferred=sum(d.action == "defer" for d in decisions),
    rejected=sum(d.action == "reject" for d in decisions),
    precision_on_accepted=accepted_correct / len(accepted) if accepted else 0.0,
    coverage=len(accepted) / total if total else 0.0,
    m_recall=m_hits / len(m_items) if m_items else 0.0,
  )
