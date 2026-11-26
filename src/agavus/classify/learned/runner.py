"""Private bounded experiment orchestration; no application activation or uploads."""

import hashlib
import importlib.metadata
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from tokenizers import Tokenizer

from agavus.classify.learned.baselines import FrozenBaseline, fit_frozen, fit_tfidf
from agavus.classify.learned.budget import choose_epochs
from agavus.classify.learned.checkpoint import (
  initialize,
  load_checkpoint,
  prepare_pretrained,
  save_checkpoint,
)
from agavus.classify.learned.comparison import (
  NeuralPredictor,
  compare,
  counterfactuals,
  cpu_latency,
  validation_loss,
)
from agavus.classify.learned.dataset import Dataset, prepare_dataset
from agavus.classify.learned.engine import Execution, encode_samples, frozen_vectors
from agavus.classify.learned.model import LearnedModel
from agavus.classify.learned.partitions import Partitions
from agavus.classify.learned.pilot import Pilot, release_memory, run_pilot
from agavus.classify.learned.report import verdict, write_report
from agavus.classify.learned.state import ExperimentState
from agavus.classify.learned.train import (
  BATCH_DOCUMENTS,
  EpochOptions,
  TokenizedSample,
  build_optimizer,
  train_epoch,
)
from agavus.classify.learned.types import SEEDS
from agavus.config.settings import Settings
from agavus.corpus.storage import CorpusStorage
from agavus.retrieve.encoder import REPO, REVISION

MAX_BUDGET_MINUTES = 120


@dataclass(frozen=True)
class RunContext:
  """One frozen snapshot and common timing bounds for every ablation."""

  state: ExperimentState
  dataset: Dataset
  encoded: tuple[TokenizedSample, ...]
  pretrained: Path
  tokenizer: Tokenizer
  execution: Execution

  def rows(self, name: str) -> tuple[TokenizedSample, ...]:
    """Read frozen membership instead of recomputing a split per model."""
    return tuple(self.encoded[i] for i in getattr(self.dataset.partitions, name))


def parameters(settings: Settings) -> dict[str, object]:
  """Record source identity, environment and all pinned experiment parameters."""
  revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True).stdout
  diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, check=True).stdout
  return {
    "git_commit": revision.decode().strip(),
    "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "encoder": REPO,
    "revision": REVISION,
    "device_requested": settings.learned_device,
    "budget_minutes": settings.learned_budget_minutes,
    "seeds": SEEDS,
    "encoder_lr": 2e-5,
    "head_lr": 1e-3,
    "optimizer": "AdamW",
    "batch_documents": BATCH_DOCUMENTS,
    "weight_decay": 0.01,
    "task_weights": [1.0, 1.0],
    "balance": "training-only",
    "window": 480,
    "overlap": 64,
    "max_epochs": 3,
    "torch_threads": torch.get_num_threads(),
    "python": sys.version,
    "versions": {
      name: importlib.metadata.version(name)
      for name in ("agavus", "numpy", "torch", "transformers", "safetensors", "tokenizers")
    },
  }


def capture_source(state: ExperimentState) -> None:
  """Only code and public build/spec files belong in the private replay archive."""
  root = Path(
    subprocess.run(
      ["git", "rev-parse", "--show-toplevel"], capture_output=True, check=True, text=True
    ).stdout.strip()
  )
  paths = [
    path
    for parent in ("src", "scripts", "stubs")
    for path in (root / parent).rglob("*")
    if path.suffix in (".py", ".pyi")
  ]
  paths.extend(root / name for name in ("pyproject.toml", "uv.lock", "docs/DESIGN.md"))
  state.capture_source(paths, root)


def ensure_private_path(data_dir: Path) -> None:
  """A configurable directory inside the public checkout must still be ignored."""
  root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
  ).stdout.strip()
  destination = data_dir.resolve() / "experiments" / "learned"
  if destination.is_relative_to(Path(root).resolve()):
    ignored = subprocess.run(
      ["git", "check-ignore", "--quiet", "--", str(destination)], check=False
    )
    if ignored.returncode:
      raise ValueError("experiment artifacts inside the repository must be gitignored")


def _fit(context: RunContext, variant: str, seed: int, epochs: int) -> tuple[Path, bool]:
  model, _ = initialize(context.pretrained, seed)
  model.to(context.execution.device)
  optimizer = build_optimizer(model)
  folder = context.state.folder / f"{variant}-{seed}"
  best = float("inf")
  history: list[dict[str, object]] = []
  completed = False
  try:
    for epoch in range(epochs):
      print(f"{variant} seed={seed} epoch={epoch + 1}/{epochs}", flush=True)
      options = EpochOptions(
        context.execution.device, seed + epoch, variant == "joint", context.execution.deadline
      )
      loss = train_epoch(model, context.rows("train"), optimizer, options)
      val = validation_loss(
        model, context.rows("validation"), context.execution, joint=variant == "joint"
      )
      history.append({"epoch": epoch + 1, "training_loss": loss, "validation_loss": val})
      if val < best:
        best = val
        save_checkpoint(model, folder, context.tokenizer)
    completed = True
  except TimeoutError:
    print(f"{variant} seed={seed}: training reserve reached", flush=True)
  context.state.put(f"history:{variant}:{seed}", history)
  del optimizer, model
  release_memory()
  return folder, completed


def _schedule(context: RunContext, pilot: Pilot, deadline: float) -> tuple[int, float, float]:
  held_windows = sum(
    len(i.windows) for name in ("calibration", "final") for i in context.rows(name)
  )
  reserve = max(
    (deadline - time.monotonic()) * 0.1, held_windows * pilot.forward_seconds * 12 + 30.0
  )
  train_windows = sum(len(i.windows) for i in context.rows("train"))
  val_windows = sum(len(i.windows) for i in context.rows("validation"))
  estimate = train_windows * pilot.train_seconds + val_windows * pilot.forward_seconds + 10.0
  remaining = max(0.001, deadline - time.monotonic())
  epochs = choose_epochs(remaining=remaining, epoch_seconds=estimate, evaluation_seconds=reserve)
  context.state.put(
    "schedule",
    {
      "common_epochs": epochs,
      "epoch_seconds_estimate": estimate,
      "evaluation_reserve_seconds": reserve,
    },
  )
  return epochs, reserve, estimate


def _frozen_model(model: LearnedModel, baseline: FrozenBaseline) -> None:
  """Convert centered numpy heads to equivalent raw-feature torch heads for CPU timing."""
  with torch.no_grad():
    for task, head in baseline.heads.items():
      target = model.area if task == "area" else model.relation
      target.weight.copy_(torch.tensor((head.model.weights * head.scale).T, dtype=torch.float32))
      target.bias.copy_(
        torch.tensor(
          head.model.bias - (head.center * head.scale) @ head.model.weights, dtype=torch.float32
        )
      )


def _evaluate_neural(
  context: RunContext,
  name: str,
  folder: Path,
  results: dict[str, dict[str, object]],
) -> None:
  print(f"calibration and locked final: {name}", flush=True)
  model, tokenizer, _ = load_checkpoint(folder)
  model.to(context.execution.device)
  predictor = NeuralPredictor(model, context.execution)
  result, temperatures = compare(predictor, context.rows("calibration"), context.rows("final"))
  result["counterfactuals"] = counterfactuals(
    model, context.rows("final"), context.execution, tokenizer
  )
  model.cpu()
  result["cpu_latency"] = cpu_latency(
    NeuralPredictor(model, Execution("cpu", context.execution.deadline)), context.rows("final")
  )
  result["temperatures"] = temperatures
  result["relation_head_trained"] = not name.startswith("area_only")
  save_checkpoint(model, folder, tokenizer, temperatures)
  results[name] = result
  context.state.put(f"result:{name}", result)
  del model, predictor
  release_memory()


def compute(context: RunContext, settings: Settings) -> dict[str, object]:
  """Pilot, identical baselines, paired ablations, then one locked final comparison."""
  deadline = context.execution.deadline
  started = deadline - settings.learned_budget_minutes * 60
  train = context.rows("train")
  print("real training pilot", flush=True)
  pilot = run_pilot(
    context.pretrained,
    context.state.folder / "pilot",
    train,
    Execution(context.execution.device, deadline - settings.learned_budget_minutes * 6),
  )
  context.state.put("pilot", asdict(pilot))
  execution = Execution(pilot.device, deadline)
  context = RunContext(
    context.state,
    context.dataset,
    context.encoded,
    context.pretrained,
    context.tokenizer,
    execution,
  )
  print(f"baselines: TF-IDF and frozen E5 on {pilot.device}", flush=True)
  tfidf = fit_tfidf(train, context.state, deadline)
  base, _ = initialize(context.pretrained, 0)
  base.to(pilot.device)
  vectors = frozen_vectors(base, context.encoded, execution)
  frozen = fit_frozen(vectors, context.dataset, context.state)
  _frozen_model(base, frozen)
  save_checkpoint(base, context.state.folder / "frozen", context.tokenizer)
  del base, vectors
  release_memory()
  epochs, reserve, estimate = _schedule(context, pilot, deadline)
  tuning = RunContext(
    context.state,
    context.dataset,
    context.encoded,
    context.pretrained,
    context.tokenizer,
    Execution(pilot.device, deadline - reserve),
  )
  jobs = [(v, s, epochs) for s in SEEDS for v in ("area_only", "joint")]
  if not epochs:
    jobs = [("joint", 0, 1)] if deadline - reserve - time.monotonic() > estimate else []
  complete = bool(epochs)
  checkpoints: list[tuple[str, Path]] = []
  for variant, seed, count in jobs:
    folder, finished = _fit(tuning, variant, seed, count)
    complete = complete and finished
    if (folder / "checkpoint.db").exists():
      checkpoints.append((f"{variant}:{seed}", folder))
    if not finished:
      break
  results: dict[str, dict[str, object]] = {}
  result, temperatures = compare(tfidf, context.rows("calibration"), context.rows("final"))
  result["cpu_latency"] = cpu_latency(tfidf, context.rows("final"))
  result["temperatures"] = temperatures
  results["tfidf"] = result
  context.state.put("result:tfidf", result)
  _evaluate_neural(context, "frozen", context.state.folder / "frozen", results)
  for name, folder in checkpoints:
    _evaluate_neural(context, name, folder, results)
  complete = complete and len(checkpoints) == len(SEEDS) * 2
  summary = verdict(results, complete=complete)
  summary.update(
    {
      "status": "complete" if complete else "partial",
      "pilot": asdict(pilot),
      "elapsed_computation_seconds": time.monotonic() - started,
      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
      * (1 if sys.platform == "darwin" else 1024),
    }
  )
  context.state.put("summary", summary)
  context.state.put("status", summary["status"])
  write_report(context.state.folder, results, summary)
  return summary


def experiment(settings: Settings) -> Path:
  """Prepare private data; insufficient independent groups stop before model computation."""
  if not 0 < settings.learned_budget_minutes <= MAX_BUDGET_MINUTES:
    raise ValueError("budget must be positive and at most 120 minutes")
  ensure_private_path(settings.data_dir)
  torch.set_num_threads(4)
  folder = (
    settings.data_dir / "experiments" / "learned" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
  )
  try:
    dataset = prepare_dataset(CorpusStorage(settings.data_dir))
  except ValueError as error:
    dataset = Dataset((), Partitions((), (), (), ()), {"error": str(error)})
    state = ExperimentState.create(folder, dataset, parameters(settings))
    state.put("status", "dataset_insufficient")
    write_report(
      folder, {}, {"status": "dataset_insufficient", "reason": str(error), "winner": None}
    )
    return folder
  pretrained = prepare_pretrained(settings.data_dir)
  state = ExperimentState.create(folder, dataset, parameters(settings))
  capture_source(state)
  started = time.monotonic()
  model, tokenizer = initialize(pretrained, 0)
  del model
  release_memory()
  encoded = encode_samples(tokenizer, dataset.samples)
  state.put("window_counts", {item.sample.id: len(item.windows) for item in encoded})
  device = settings.learned_device
  if device == "auto":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
  if device not in ("cpu", "mps"):
    raise ValueError("device must be auto, cpu or mps")
  context = RunContext(
    state,
    dataset,
    encoded,
    pretrained,
    tokenizer,
    Execution(device, started + settings.learned_budget_minutes * 60),
  )
  try:
    compute(context, settings)
  except (TimeoutError, RuntimeError, ValueError) as error:
    summary: dict[str, object] = {
      "status": "partial",
      "winner": None,
      "reason": str(error),
      "elapsed_computation_seconds": time.monotonic() - started,
    }
    state.put("summary", summary)
    state.put("status", "partial")
    write_report(folder, state.results(), summary)
  return folder
