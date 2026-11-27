"""Shared training, run on both PyTorch models (synthetic text)."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import pytest
import torch

from agavus.model import agathe, neural
from agavus.model.agathe import AgatheModel
from agavus.model.bigram import build_vocab, train_bigram, uniform_loss
from agavus.model.data import Split
from agavus.model.neural import NeuralModel
from agavus.model.tokenizer import train_tokenizer
from agavus.model.training import (
  LOG_EVERY,
  History,
  Init,
  Progress,
  check_sample,
  descend,
  synchronize,
)

# Many threads only fight over work this small.
torch.set_num_threads(1)

TEXTS = ("guarda guarda", "a guarda do filho")
STEPS = 200
NEURAL = neural.NeuralConfig(context=3, vector=8, hidden=32, batch=32, rate=1e-2, check_every=50)
AGATHE = agathe.AgatheConfig(
  context=8, width=16, heads=2, kv_heads=1, layers=1, batch=16, rate=1e-2, check_every=50
)
VOCAB = build_vocab(TEXTS)
TOKENIZER = train_tokenizer(TEXTS * 3, 300)

Model = NeuralModel | AgatheModel
Trainer = Callable[[int, Sequence[str]], tuple[Model, History]]


def _neural(steps: int, validation: Sequence[str]) -> tuple[Model, History]:
  return neural.train(Split(TEXTS, tuple(validation)), VOCAB, NEURAL, steps=steps)


def _agathe(steps: int, validation: Sequence[str]) -> tuple[Model, History]:
  return agathe.train(Split(TEXTS, tuple(validation)), TOKENIZER, AGATHE, steps=steps)


def _nats_per_char(model: Model) -> float:
  """Surprise per character, so a BPE model compares with the character bigram."""
  if isinstance(model, AgatheModel):
    return model.bits_per_char(TEXTS) * math.log(2)
  return model.loss(TEXTS)


TRAINERS = pytest.mark.parametrize("trainer", [_neural, _agathe], ids=["neural", "agathe"])


def test_init_is_reproducible_and_biases_start_at_zero() -> None:
  first, second = Init.seeded(3, "cpu"), Init.seeded(3, "cpu")
  assert torch.equal(first.matrix(2, 3, 1.0), second.matrix(2, 3, 1.0))
  assert first.zeros(4).tolist() == [0.0] * 4


def test_untrained_models_are_near_uniform() -> None:
  assert neural.start(VOCAB, NEURAL).loss(TEXTS) == pytest.approx(uniform_loss(VOCAB), abs=0.1)
  blank = agathe.start(TOKENIZER, AGATHE).loss(TEXTS)
  assert blank == pytest.approx(math.log(TOKENIZER.get_vocab_size()), abs=0.1)


@TRAINERS
def test_learns_more_than_the_bigram(trainer: Trainer) -> None:
  model, history = trainer(STEPS, TEXTS)
  assert _nats_per_char(model) < train_bigram(TEXTS).loss(TEXTS)
  assert [number for number, _, _ in history] == [50, 100, 150, 200]


@TRAINERS
def test_no_validation_means_no_history(trainer: Trainer) -> None:
  _, history = trainer(STEPS // 4, ())
  assert history == []


@TRAINERS
def test_training_and_sampling_are_reproducible(trainer: Trainer) -> None:
  first, _ = trainer(STEPS // 4, ())
  second, _ = trainer(STEPS // 4, ())
  assert math.isclose(first.loss(TEXTS), second.loss(TEXTS), rel_tol=1e-4)
  sample = first.sample(seed=1, max_chars=20)
  assert sample == first.sample(seed=1, max_chars=20)
  assert len(sample) <= len("x" * 20)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires local MPS")
def test_mps_training_validation_and_sampling() -> None:
  split = Split(TEXTS, TEXTS)
  for model, history in (
    neural.train(split, VOCAB, replace(NEURAL, device="mps", check_every=1), steps=1),
    agathe.train(split, TOKENIZER, replace(AGATHE, device="mps", check_every=1), steps=1),
  ):
    synchronize("mps")
    assert all(weight.device.type == "mps" for weight in model.parameters())
    assert math.isfinite(history[0][2])
    assert len(model.sample(seed=1, max_chars=2)) <= len("ab")


def test_check_sample_keeps_small_text_and_spaces_out_large_text() -> None:
  small = ("ab", "cd")
  assert check_sample(small, chars=10) == small
  large = tuple(f"{index:04d}" for index in range(100))
  sample = check_sample(large, chars=40)
  assert sample == large[::10]
  assert sum(map(len, sample)) == len("0000") * 10


@dataclass(frozen=True)
class _Steady:
  """Settings for a short descent: checks every 150 steps, progress every LOG_EVERY."""

  rate: float = 0.01
  weight_decay: float = 0.0
  check_every: int = 150
  log_every: int = LOG_EVERY
  seed: int = 0


def test_descent_prints_progress_every_log_step_check_and_the_end(
  capsys: pytest.CaptureFixture[str],
) -> None:
  weight = torch.zeros(2, requires_grad=True)

  def mistake(_: torch.Generator) -> torch.Tensor:
    return ((weight - 1.0) ** 2).sum()

  steps = 2 * LOG_EVERY + 50
  history = descend([weight], mistake, _Steady(), steps=steps, check=lambda: (0.5, 0.25))
  lines = capsys.readouterr().out.splitlines()
  assert [line.split()[1] for line in lines] == [
    f"{n}/{steps}" for n in (LOG_EVERY, 150, 2 * LOG_EVERY, steps)
  ]
  assert [line.endswith("train 0.5000  held-out 0.2500") for line in lines] == [
    False,
    True,
    False,
    False,
  ]
  assert history == [(150, 0.5, 0.25)]
  assert float(lines[-1].split()[3]) < float(lines[0].split()[3])
  silent = replace(_Steady(), log_every=0)
  assert descend([weight], mistake, silent, steps=3, check=None) == []
  assert capsys.readouterr().out == ""


def test_progress_line_shows_rate_eta_and_checks() -> None:
  line = Progress(1200, 12000, 0.41234, 216.0, (0.3, 0.42)).line()
  assert line == (
    "step   1200/12000  batch 0.4123  0.180 s/step  eta 0h32m  train 0.3000  held-out 0.4200"
  )
  assert Progress(1, 2, 1.0, 7200.0).line().endswith("eta 2h00m")
