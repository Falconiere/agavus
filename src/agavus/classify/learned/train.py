"""Explicit AdamW updates with equal document and task contributions."""

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from agavus.classify.learned.model import LearnedModel, tensors
from agavus.classify.learned.types import Sample

BATCH_DOCUMENTS = 16


@dataclass(frozen=True)
class TokenizedSample:
  """A document is one observation even when it has many windows."""

  sample: Sample
  windows: tuple[tuple[int, ...], ...]
  target: int


def check_deadline(deadline: float) -> None:
  """Stop cooperatively between bounded encoder operations."""
  if time.monotonic() >= deadline:
    raise TimeoutError("local computation budget exhausted")


def build_optimizer(model: LearnedModel) -> torch.optim.AdamW:
  """Pinned encoder/head learning rates, with no inference-time adaptation."""
  return torch.optim.AdamW(
    [
      {"params": model.encoder.parameters(), "lr": 2e-5},
      {"params": [*model.area.parameters(), *model.relation.parameters()], "lr": 1e-3},
    ]
  )


@dataclass(frozen=True)
class StepOptions:
  """One update's execution bounds and training-only weight."""

  device: str = "cpu"
  weight: float = 1.0
  deadline: float = float("inf")


@dataclass(frozen=True)
class EpochOptions:
  """Fixed random order and task ablation, independent of held-out examples."""

  device: str
  seed: int
  joint: bool
  deadline: float


def train_step(
  model: LearnedModel,
  example: TokenizedSample,
  optimizer: torch.optim.Optimizer,
  options: StepOptions = StepOptions(),
) -> float:
  """One optimizer update per document; average window CE prevents length bias."""
  optimizer.zero_grad(set_to_none=True)
  total = _backprop(model, example, options)
  _update(model, optimizer)
  return total


def _update(model: LearnedModel, optimizer: torch.optim.Optimizer) -> None:
  nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  optimizer.step()


def _backprop(model: LearnedModel, example: TokenizedSample, options: StepOptions) -> float:
  windows = example.windows
  if not windows:
    raise ValueError("need nonempty windows")
  model.train()
  total = 0.0
  for window in windows:
    check_deadline(options.deadline)
    logits = model.logits(*tensors([window], options.device), example.sample.task)
    loss = functional.cross_entropy(logits, torch.tensor([example.target], device=options.device))
    scaled = loss * (options.weight / len(windows))
    if not bool(torch.isfinite(scaled)):
      raise ValueError("nonfinite training loss")
    torch.autograd.backward(scaled)
    total += float(scaled.detach().cpu())
  return total


def train_epoch(
  model: LearnedModel,
  examples: Sequence[TokenizedSample],
  optimizer: torch.optim.Optimizer,
  options: EpochOptions,
) -> float:
  """Balance classes from training only; each task sums to the same weight."""
  selected = [item for item in examples if options.joint or item.sample.task == "area"]
  counts = Counter((item.sample.task, item.target) for item in selected)
  task_sizes = Counter(item.sample.task for item in selected)
  task_classes = Counter(task for task, _ in counts)
  losses: list[float] = []
  if not selected:
    raise ValueError("need nonempty training documents")
  order = np.random.default_rng(options.seed).permutation(len(selected))
  for start in range(0, len(order), BATCH_DOCUMENTS):
    batch = order[start : start + BATCH_DOCUMENTS]
    optimizer.zero_grad(set_to_none=True)
    for position in batch:
      item = selected[int(position)]
      task = item.sample.task
      # Equal task mass despite unequal document counts; windows share their document's mass.
      weight = len(selected) / (len(task_sizes) * task_classes[task] * counts[task, item.target])
      step = StepOptions(options.device, weight / len(batch), options.deadline)
      losses.append(_backprop(model, item, step) * len(batch))
    _update(model, optimizer)
    if start % (BATCH_DOCUMENTS * BATCH_DOCUMENTS) == 0:
      print(f"  documents {min(start + len(batch), len(order))}/{len(order)}", flush=True)
  return sum(losses) / len(losses)
