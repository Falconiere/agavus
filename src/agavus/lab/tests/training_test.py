"""Shared training, run on both PyTorch models (synthetic text)."""

import math
from collections.abc import Callable, Sequence

import pytest
import torch

from agavus.lab import gpt, neural
from agavus.lab.bigram import Vocab, build_vocab, train_bigram, uniform_loss
from agavus.lab.data import Split
from agavus.lab.gpt import GptModel
from agavus.lab.neural import NeuralModel
from agavus.lab.training import History, Init

# Many threads only fight over work this small.
torch.set_num_threads(1)

TEXTS = ("guarda guarda", "a guarda do filho")
STEPS = 200
NEURAL = neural.NeuralConfig(context=3, vector=8, hidden=32, batch=32, rate=1e-2, check_every=50)
GPT = gpt.GptConfig(context=8, width=16, heads=2, layers=1, batch=16, rate=1e-2, check_every=50)

Model = NeuralModel | GptModel
Trainer = Callable[[Vocab, int, Sequence[str]], tuple[Model, History]]


def _neural(vocab: Vocab, steps: int, validation: Sequence[str]) -> tuple[Model, History]:
  return neural.train(Split(TEXTS, tuple(validation)), vocab, NEURAL, steps=steps)


def _gpt(vocab: Vocab, steps: int, validation: Sequence[str]) -> tuple[Model, History]:
  return gpt.train(Split(TEXTS, tuple(validation)), vocab, GPT, steps=steps)


TRAINERS = pytest.mark.parametrize("trainer", [_neural, _gpt], ids=["neural", "gpt"])


def test_init_is_reproducible_and_biases_start_at_zero() -> None:
  first, second = Init.seeded(3, "cpu"), Init.seeded(3, "cpu")
  assert torch.equal(first.matrix(2, 3, 1.0), second.matrix(2, 3, 1.0))
  assert first.zeros(4).tolist() == [0.0] * 4


def test_untrained_models_are_near_uniform() -> None:
  vocab = build_vocab(TEXTS)
  for model in (neural.start(vocab, NEURAL), gpt.start(vocab, GPT)):
    assert model.loss(TEXTS) == pytest.approx(uniform_loss(vocab), abs=0.1)


@TRAINERS
def test_learns_more_than_the_bigram(trainer: Trainer) -> None:
  vocab = build_vocab(TEXTS)
  model, history = trainer(vocab, STEPS, TEXTS)
  assert model.loss(TEXTS) < train_bigram(TEXTS).loss(TEXTS)
  assert [number for number, _, _ in history] == [50, 100, 150, 200]


@TRAINERS
def test_no_validation_means_no_history(trainer: Trainer) -> None:
  _, history = trainer(build_vocab(TEXTS), STEPS // 4, ())
  assert history == []


@TRAINERS
def test_training_and_sampling_are_reproducible(trainer: Trainer) -> None:
  vocab = build_vocab(TEXTS)
  first, _ = trainer(vocab, STEPS // 4, ())
  second, _ = trainer(vocab, STEPS // 4, ())
  assert math.isclose(first.loss(TEXTS), second.loss(TEXTS), rel_tol=1e-4)
  sample = first.sample(seed=1, max_chars=20)
  assert sample == first.sample(seed=1, max_chars=20)
  assert len(sample) <= len("x" * 20)
