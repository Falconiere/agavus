"""Small neural language model: several characters back (docs/llm-lab.md §4).

Each character becomes a learned vector; the vectors of the last few
characters go through one hidden layer, which scores every next character.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from agavus.language_model.bigram import BOUNDARY, Vocab
from agavus.language_model.data import Split
from agavus.language_model.training import History, Init, fit, mean_surprise, pick

EVAL_BATCH = 8192


@dataclass(frozen=True)
class NeuralConfig:
  """Size of the model and how it trains."""

  context: int = 4
  vector: int = 24
  hidden: int = 256
  batch: int = 256
  rate: float = 2e-3
  weight_decay: float = 1e-2
  check_every: int = 250
  seed: int = 0
  device: str = "cpu"


@dataclass(frozen=True)
class NeuralModel:
  """Weights: character vectors, hidden layer, output layer."""

  config: NeuralConfig
  vocab: Vocab
  vectors: Tensor
  hidden_weights: Tensor
  hidden_bias: Tensor
  out_weights: Tensor
  out_bias: Tensor

  def parameters(self) -> list[Tensor]:
    """Everything that training adjusts."""
    return [
      self.vectors,
      self.hidden_weights,
      self.hidden_bias,
      self.out_weights,
      self.out_bias,
    ]

  def scores(self, contexts: Tensor) -> Tensor:
    """One score per next character for every context."""
    looked_up = self.vectors[contexts].flatten(1)
    hidden = torch.tanh(looked_up @ self.hidden_weights + self.hidden_bias)
    return hidden @ self.out_weights + self.out_bias

  def loss(self, texts: Sequence[str]) -> float:
    """Mean surprise over every character, like the bigram's."""
    contexts, answers = examples(texts, self.vocab, self.config)
    return mean_surprise(self.scores, contexts, answers, EVAL_BATCH)

  def sample(self, *, seed: int, max_chars: int) -> str:
    """Write one character at a time until a boundary or max_chars."""
    generator = torch.Generator().manual_seed(seed)
    recent = [BOUNDARY] * self.config.context
    ids: list[int] = []
    while len(ids) < max_chars:
      context = torch.tensor([recent], device=self.config.device)
      chosen = pick(self.scores(context)[0], generator)
      if chosen == BOUNDARY:
        break
      ids.append(chosen)
      recent = [*recent[1:], chosen]
    return self.vocab.decode(ids)


def examples(texts: Sequence[str], vocab: Vocab, config: NeuralConfig) -> tuple[Tensor, Tensor]:
  """(previous characters, next character) for every position of every text."""
  contexts: list[list[int]] = []
  answers: list[int] = []
  for text in texts:
    ids = vocab.encode(text)
    padded = [BOUNDARY] * (config.context - 1) + ids
    for position in range(len(ids) - 1):
      contexts.append(padded[position : position + config.context])
      answers.append(ids[position + 1])
  return (
    torch.tensor(contexts, device=config.device),
    torch.tensor(answers, device=config.device),
  )


def start(vocab: Vocab, config: NeuralConfig) -> NeuralModel:
  """Random small weights, reproducible by seed."""
  init = Init.seeded(config.seed, config.device)
  inputs = config.context * config.vector
  return NeuralModel(
    config=config,
    vocab=vocab,
    vectors=init.matrix(vocab.size, config.vector, 1.0),
    hidden_weights=init.matrix(inputs, config.hidden, 1.0 / math.sqrt(inputs)),
    hidden_bias=init.zeros(config.hidden),
    out_weights=init.matrix(config.hidden, vocab.size, 0.01),
    out_bias=init.zeros(vocab.size),
  )


def train(
  split: Split, vocab: Vocab, config: NeuralConfig, *, steps: int
) -> tuple[NeuralModel, History]:
  """Learn from split.train; history is checked on split.holdout, if any (see `fit`)."""
  model = start(vocab, config)
  contexts, answers = examples(split.train, vocab, config)

  def mistake(generator: torch.Generator) -> Tensor:
    picked = torch.randint(len(answers), (config.batch,), generator=generator).to(config.device)
    return cross_entropy(model.scores(contexts[picked]), answers[picked])

  return model, fit(model, mistake, config, steps=steps, watch=split)
