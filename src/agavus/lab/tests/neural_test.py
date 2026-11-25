"""Neural language model: examples, learning, determinism (synthetic text)."""

import math

import pytest
import torch

from agavus.lab.bigram import BOUNDARY, build_vocab, train_bigram, uniform_loss
from agavus.lab.neural import NeuralConfig, examples, start, train

# Many threads only fight over work this small.
torch.set_num_threads(1)

TEXTS = ["guarda guarda", "a guarda do filho"]
CONFIG = NeuralConfig(context=3, vector=8, hidden=32, batch=32, rate=1e-2, check_every=50)
STEPS = 200


def test_examples_look_back_and_pad() -> None:
  vocab = build_vocab(["ab"])
  contexts, answers = examples(["ab"], vocab, CONFIG)
  a, b = vocab.encode("ab")[1:3]
  assert contexts.tolist() == [[BOUNDARY] * 3, [BOUNDARY, BOUNDARY, a], [BOUNDARY, a, b]]
  assert answers.tolist() == [a, b, BOUNDARY]


def test_examples_count_matches_the_bigram() -> None:
  _, answers = examples(TEXTS, build_vocab(TEXTS), CONFIG)
  assert len(answers) == sum(len(text) + 1 for text in TEXTS)


def test_untrained_model_is_near_uniform() -> None:
  vocab = build_vocab(TEXTS)
  assert start(vocab, CONFIG).loss(TEXTS) == pytest.approx(uniform_loss(vocab), abs=0.1)


def test_learns_more_than_the_bigram() -> None:
  vocab = build_vocab(TEXTS)
  model, history = train(TEXTS, vocab, CONFIG, steps=STEPS, validation=TEXTS)
  assert model.loss(TEXTS) < train_bigram(TEXTS).loss(TEXTS)
  assert [number for number, _, _ in history] == [50, 100, 150, 200]


def test_training_and_sampling_are_reproducible() -> None:
  vocab = build_vocab(TEXTS)
  first, _ = train(TEXTS, vocab, CONFIG, steps=STEPS // 4)
  second, _ = train(TEXTS, vocab, CONFIG, steps=STEPS // 4)
  assert math.isclose(first.loss(TEXTS), second.loss(TEXTS), rel_tol=1e-4)
  sample = first.sample(seed=1, max_chars=20)
  assert sample == first.sample(seed=1, max_chars=20)
  assert len(sample) <= len("x" * 20)
