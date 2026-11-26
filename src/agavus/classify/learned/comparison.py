"""Calibrate once and score all locked comparisons after checkpoint selection."""

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from tokenizers import Tokenizer

from agavus.classify.calibrate import fit_temperature
from agavus.classify.learned.baselines import Predictor
from agavus.classify.learned.engine import Execution, predict_logits, task_classes
from agavus.classify.learned.metrics import area_metrics, relation_metrics
from agavus.classify.learned.model import LearnedModel
from agavus.classify.learned.train import TokenizedSample
from agavus.classify.learned.windows import document_windows
from agavus.classify.model import FloatArray


@dataclass(frozen=True)
class NeuralPredictor:
  """Adapt the explicit encoder forward pass to the shared score-only interface."""

  model: LearnedModel
  execution: Execution

  def logits(self, examples: Sequence[TokenizedSample]) -> FloatArray:
    """No updates, generation or production side effects."""
    return predict_logits(self.model, examples, self.execution)


def compare(
  predictor: Predictor,
  calibration: Sequence[TokenizedSample],
  final: Sequence[TokenizedSample],
) -> tuple[dict[str, object], tuple[float, float]]:
  """Temperatures see calibration only; final labels enter metrics only."""
  report: dict[str, object] = {}
  temperatures: list[float] = []
  for task, measure in (("area", area_metrics), ("relation", relation_metrics)):
    cal = [i for i in calibration if i.sample.task == task]
    held = [i for i in final if i.sample.task == task]
    temperature = fit_temperature(predictor.logits(cal), [i.target for i in cal])
    report[task] = measure(
      [i.sample for i in held], predictor.logits(held), task_classes(task), temperature
    )
    temperatures.append(temperature)
  return report, (temperatures[0], temperatures[1])


def validation_loss(
  model: LearnedModel,
  examples: Sequence[TokenizedSample],
  execution: Execution,
  *,
  joint: bool,
) -> float:
  """Equal task-mean NLL selects checkpoints; calibration remains untouched."""
  losses: list[float] = []
  for task in ("area", "relation") if joint else ("area",):
    selected = [i for i in examples if i.sample.task == task]
    scores = predict_logits(model, selected, execution)
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_partition = np.log(np.exp(shifted).sum(axis=1))
    losses.append(
      float(
        (log_partition - shifted[np.arange(len(selected)), [i.target for i in selected]]).mean()
      )
    )
  return sum(losses) / len(losses)


def cpu_latency(predictor: Predictor, examples: Sequence[TokenizedSample]) -> dict[str, float]:
  """Warm and time full documents on CPU; window counts remain visible in the snapshot."""
  selected = [i for i in examples if i.sample.task == "area"][:3]
  predictor.logits(selected[:1])
  durations: list[float] = []
  for item in selected:
    start = time.monotonic()
    predictor.logits([item])
    durations.append(time.monotonic() - start)
  return {
    "median_seconds": float(np.median(durations)),
    "max_seconds": max(durations),
    "n_documents": float(len(selected)),
  }


def counterfactuals(
  model: LearnedModel,
  examples: Sequence[TokenizedSample],
  execution: Execution,
  tokenizer: Tokenizer,
) -> dict[str, object]:
  """Measure grant/denial relation flips and area stability on unseen families."""
  groups: dict[str, list[TokenizedSample]] = {}
  for item in examples:
    if item.sample.task == "relation":
      groups.setdefault(item.sample.group, []).append(item)
  failures: list[str] = []
  for group, members in groups.items():
    positive = next(i for i in members if i.sample.id.endswith(":0"))
    negative = next(i for i in members if i.sample.id.endswith(":2"))
    pair = [positive, negative]
    relation = predict_logits(model, pair, execution).argmax(axis=1).tolist()
    area_pair = [
      TokenizedSample(
        replace(item.sample, statement=""), document_windows(tokenizer, item.sample.text), 0
      )
      for item in pair
    ]
    area_rows = predict_logits(model, area_pair, execution).argmax(axis=1).tolist()
    if relation != [0, 1] or area_rows[0] != area_rows[1]:
      failures.append(group)
  return {"families": len(groups), "relation_flip_and_area_stable_failures": failures}
