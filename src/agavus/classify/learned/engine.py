"""Tokenization and bounded forward passes shared by all encoder experiments."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from tokenizers import Tokenizer

from agavus.classify.learned.model import LearnedModel, tensors
from agavus.classify.learned.train import TokenizedSample, check_deadline
from agavus.classify.learned.types import RELATIONS, Sample
from agavus.classify.learned.windows import document_windows, relation_window
from agavus.classify.model import FloatArray
from agavus.taxonomy.areas import AREAS


@dataclass(frozen=True)
class Execution:
  """Local device and a monotonic wall-time ceiling, checked between windows."""

  device: str
  deadline: float


def task_classes(task: str) -> tuple[str, ...]:
  """Experimental relation labels never extend the public registry."""
  return tuple(sorted(AREAS)) if task == "area" else RELATIONS


def encode_samples(tokenizer: Tokenizer, samples: Sequence[Sample]) -> tuple[TokenizedSample, ...]:
  """No synonym normalization, stopword removal, cue extraction or truncation."""
  result: list[TokenizedSample] = []
  for sample in samples:
    windows = (
      (relation_window(tokenizer, sample.text, sample.statement),)
      if sample.statement
      else document_windows(tokenizer, sample.text)
    )
    result.append(TokenizedSample(sample, windows, task_classes(sample.task).index(sample.label)))
  return tuple(result)


def predict_logits(
  model: LearnedModel,
  examples: Sequence[TokenizedSample],
  execution: Execution,
) -> FloatArray:
  """Each document contributes one mean-logit row, including its last window."""
  return _forward(model, examples, execution, head=True)


def frozen_vectors(
  encoder: LearnedModel,
  documents: Sequence[TokenizedSample],
  runtime: Execution,
) -> FloatArray:
  """Average raw pretrained window features; fitted normalization uses training only."""
  return _forward(encoder, documents, runtime, head=False)


def _forward(
  model: LearnedModel,
  examples: Sequence[TokenizedSample],
  execution: Execution,
  *,
  head: bool,
) -> FloatArray:
  model.eval()
  output: list[list[float]] = []
  with torch.no_grad():
    for item in examples:
      rows = []
      for window in item.windows:
        check_deadline(execution.deadline)
        inputs = tensors([window], execution.device)
        value = model.logits(*inputs, item.sample.task) if head else model.features(*inputs)
        rows.append(value)
      output.append(torch.cat(rows).mean(dim=0).cpu().tolist())
  return np.asarray(output, dtype=np.float64)
