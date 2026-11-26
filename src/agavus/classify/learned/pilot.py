"""Real local training pilot; MPS is an optimization, never a requirement."""

import gc
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from agavus.classify.learned.checkpoint import initialize, save_checkpoint
from agavus.classify.learned.engine import Execution, predict_logits
from agavus.classify.learned.train import (
  BATCH_DOCUMENTS,
  EpochOptions,
  TokenizedSample,
  build_optimizer,
  train_epoch,
)


@dataclass(frozen=True)
class Pilot:
  """Conservative per-window times include backward and optimizer work."""

  device: str
  train_seconds: float
  forward_seconds: float
  fallback: str


def release_memory() -> None:
  """Discard each ablation before loading the next full encoder."""
  gc.collect()
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()


def _pilot(
  folder: Path,
  output: Path,
  examples: Sequence[TokenizedSample],
  execution: Execution,
) -> Pilot:
  model, tokenizer = initialize(folder, 0)
  model.to(execution.device)
  optimizer = build_optimizer(model)
  first = next(model.encoder.parameters())
  before = first.detach().cpu().clone()
  picked = list(examples[:: max(1, len(examples) // BATCH_DOCUMENTS)][:BATCH_DOCUMENTS])
  picked[0] = max(examples, key=lambda e: len(e.windows))
  window_count = sum(len(item.windows) for item in picked)
  start = time.monotonic()
  train_epoch(model, picked, optimizer, EpochOptions(execution.device, 0, True, execution.deadline))
  training = (time.monotonic() - start) / window_count
  start = time.monotonic()
  scores = predict_logits(model, picked, execution)
  forward = (time.monotonic() - start) / window_count
  if not bool(torch.isfinite(torch.from_numpy(scores)).all()):
    raise RuntimeError("pilot produced nonfinite logits")
  if torch.equal(before, first.detach().cpu()):
    raise RuntimeError("pilot did not change encoder weights")
  save_checkpoint(model, output, tokenizer)
  return Pilot(execution.device, training, forward, "")


def run_pilot(
  folder: Path,
  output: Path,
  examples: Sequence[TokenizedSample],
  execution: Execution,
) -> Pilot:
  """Auto's MPS proposal falls back only after a failed real training operation."""
  fallback = ""
  result: Pilot | None
  try:
    result = _pilot(folder, output, examples, execution)
  except (RuntimeError, NotImplementedError) as error:
    if execution.device != "mps":
      raise
    fallback = f"MPS pilot failed: {type(error).__name__}: {error}"
    result = None
  if result is None:
    release_memory()
    result = _pilot(folder, output, examples, Execution("cpu", execution.deadline))
  release_memory()
  return Pilot(result.device, result.train_seconds, result.forward_seconds, fallback)
