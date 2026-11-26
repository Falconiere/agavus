"""Small real CPU integration through baselines, pilot, calibration and command guards."""

import copy
import time
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from transformers import PreTrainedModel

from agavus.classify.learned.baselines import fit_frozen, fit_tfidf
from agavus.classify.learned.comparison import NeuralPredictor, compare, validation_loss
from agavus.classify.learned.dataset import Dataset
from agavus.classify.learned.engine import Execution, encode_samples
from agavus.classify.learned.partitions import Partitions
from agavus.classify.learned.pilot import run_pilot
from agavus.classify.learned.runner import ensure_private_path, experiment
from agavus.classify.learned.state import ExperimentState
from agavus.classify.learned.train import TokenizedSample, build_optimizer, train_step
from agavus.classify.learned.types import RELATIONS, Sample
from agavus.classify.store import load_model
from agavus.config.settings import Settings
from agavus.taxonomy.areas import AREAS

from .fixtures import tiny_model, tiny_tokenizer

CALIBRATION_TEMPERATURES = (1.7, 2.3)


def _dataset() -> Dataset:
  samples = tuple(
    Sample(
      f"{split}:{task}:{label}",
      f"pedido {label} {'xenolitounseen' if split else 'negado'}",
      label,
      f"{split}:{task}:{label}",
      "synthetic:fixture",
      "authored",
      statement="pedido sim" if task == "relation" else "",
      category="direct_negation",
    )
    for split in range(3)
    for task, labels in (("area", tuple(AREAS)), ("relation", RELATIONS))
    for label in labels
  )
  indexes = tuple(range(6))
  return Dataset(samples, Partitions(indexes, (), tuple(range(6, 12)), tuple(range(12, 18))), {})


def test_baselines_fit_training_only_and_calibration_ignores_final_labels(tmp_path: Path) -> None:
  dataset = _dataset()
  encoded = encode_samples(tiny_tokenizer(), dataset.samples)
  state = ExperimentState.create(tmp_path / "run", dataset, {})
  training = encoded[:6]
  tfidf = fit_tfidf(training, state, time.monotonic() + 30)
  assert all("xenolitounseen" not in c.vectorizer.vocab for c in tfidf.models.values())
  vectors = np.random.default_rng(0).normal(size=(len(encoded), 8))
  frozen = fit_frozen(vectors, dataset, state)
  for task, head in frozen.heads.items():
    indexes = [i for i in dataset.partitions.train if dataset.samples[i].task == task]
    assert np.allclose(head.center, vectors[indexes].mean(axis=0))
  assert np.isfinite(frozen.logits(encoded[:3])).all()
  execution = Execution("cpu", time.monotonic() + 30)
  predictor = NeuralPredictor(tiny_model(), execution)
  calibration, final = encoded[6:12], encoded[12:]
  _, first = compare(predictor, calibration, final)
  changed = tuple(
    replace(
      item,
      target=(item.target + 1) % 3,
      sample=replace(
        item.sample,
        label=(RELATIONS if item.sample.statement else tuple(sorted(AREAS)))[(item.target + 1) % 3],
      ),
    )
    for item in final
  )
  _, second = compare(predictor, calibration, changed)
  assert first == second
  assert np.isfinite(validation_loss(predictor.model, calibration, execution, joint=True))
  assert (state.folder / "baselines.db").is_file()
  tfidf.save_calibrated(state.folder, CALIBRATION_TEMPERATURES)
  for task, temperature in zip(("area", "relation"), CALIBRATION_TEMPERATURES, strict=True):
    assert load_model(state.folder / "baselines.db", f"tfidf:{task}").temperature == temperature


def test_real_cpu_pilot_and_document_length_normalization(tmp_path: Path) -> None:
  model = tiny_model()
  second = copy.deepcopy(model)
  sample = Sample("id", "pedido", next(iter(sorted(AREAS))), "g", "synthetic:one", "authored")
  single = TokenizedSample(sample, ((2, 4, 3),), 0)
  repeated = replace(single, windows=single.windows * 2)
  first_loss = train_step(model, single, build_optimizer(model))
  other_loss = train_step(second, repeated, build_optimizer(second))
  assert first_loss == pytest.approx(other_loss)
  assert torch.allclose(next(model.encoder.parameters()), next(second.encoder.parameters()))
  source = tmp_path / "pretrained"
  cast(PreTrainedModel, model.encoder).save_pretrained(str(source), safe_serialization=True)
  tiny_tokenizer().save(str(source / "tokenizer.json"))
  samples = encode_samples(tiny_tokenizer(), _dataset().samples[:6])
  pilot = run_pilot(source, tmp_path / "pilot", samples, Execution("cpu", time.monotonic() + 30))
  assert pilot.device == "cpu" and pilot.train_seconds > 0 and pilot.forward_seconds > 0
  assert (tmp_path / "pilot" / "heads.safetensors").is_file()


def test_budget_guard_stops_before_any_data_or_download(tmp_path: Path) -> None:
  settings = Settings("test", tmp_path, date(2020, 1, 1), "", learned_budget_minutes=121.0)
  with pytest.raises(ValueError, match="at most 120"):
    experiment(settings)
  assert list(tmp_path.iterdir()) == []
  ensure_private_path(tmp_path)
  with pytest.raises(ValueError, match="must be gitignored"):
    ensure_private_path(Path.cwd() / "unignored-private-experiment")
