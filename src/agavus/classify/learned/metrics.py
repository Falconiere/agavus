"""Locked final-partition measurements; never checkpoint-selection criteria."""

from collections.abc import Sequence

import numpy as np

from agavus.classify.calibrate import expected_calibration_error
from agavus.classify.learned.types import Sample
from agavus.classify.model import FloatArray, softmax
from agavus.decide.losses import accept_action
from agavus.decide.rule import decide
from agavus.evals.goldset import GoldItem
from agavus.evals.metrics import macro_f1
from agavus.evals.routing import evaluate_routing
from agavus.taxonomy.areas import MARIA_DA_PENHA

M_RECALL_TARGET = 0.98
ECE_TARGET = 0.05
PRECISION_TARGET = 0.95


def _predictions(logits: FloatArray, classes: Sequence[str], temperature: float) -> list[str]:
  return [classes[int(index)] for index in (logits / temperature).argmax(axis=1)]


def area_metrics(
  samples: Sequence[Sample],
  logits: FloatArray,
  classes: Sequence[str],
  temperature: float,
) -> dict[str, object]:
  """Document macro-F1, calibrated decision quality, and explicit unsafe M ids."""
  probabilities = softmax(logits / temperature)
  mappings = [
    {label: float(value) for label, value in zip(classes, row, strict=True)}
    for row in probabilities
  ]
  items = [GoldItem(s.id, s.text, s.label) for s in samples]
  routing = evaluate_routing(items, mappings)
  decisions = [decide(probs).action for probs in mappings]
  accepted = [i for i, action in enumerate(decisions) if action.startswith("accept-")]
  correct = sum(decisions[i] == accept_action(samples[i].label) for i in accepted)
  predictions = _predictions(logits, classes, temperature)
  targets = [classes.index(s.label) for s in samples]
  return {
    "n": len(samples),
    "macro_f1": macro_f1(predictions, [s.label for s in samples], classes),
    "m_recall": routing.m_hits / routing.m_total if routing.m_total else 0.0,
    "m_recall_target_met": bool(
      routing.m_total and routing.m_hits / routing.m_total >= M_RECALL_TARGET
    ),
    "unsafe_m_decisions": list(routing.unsafe),
    "m_misroutes": [
      s.id
      for s, p in zip(samples, predictions, strict=True)
      if s.label == MARIA_DA_PENHA.code and p != s.label
    ],
    "ece": expected_calibration_error(probabilities, targets),
    "accepted_precision": correct / len(accepted) if accepted else 0.0,
    "coverage": len(accepted) / len(samples),
    "accepted": len(accepted),
    "probabilities": mappings,
  }


def relation_metrics(
  pairs: Sequence[Sample],
  scores: FloatArray,
  relations: Sequence[str],
  temperature: float,
) -> dict[str, object]:
  """Count failures by unseen scenario category, retaining ids for audit."""
  predicted = _predictions(scores, relations, temperature)
  failures: dict[str, list[str]] = {s.category: [] for s in pairs}
  for sample, label in zip(pairs, predicted, strict=True):
    if sample.label != label:
      failures[sample.category].append(sample.id)
  return {
    "n": len(pairs),
    "macro_f1": macro_f1(predicted, [s.label for s in pairs], relations),
    "category_failures": failures,
    "probabilities": softmax(scores / temperature).tolist(),
    "finite": bool(np.isfinite(scores).all()),
  }
