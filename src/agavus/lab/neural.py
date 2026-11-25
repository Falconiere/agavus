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

from agavus.lab.bigram import BOUNDARY, Vocab

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
    total = 0.0
    with torch.no_grad():
      for start in range(0, len(answers), EVAL_BATCH):
        part = slice(start, start + EVAL_BATCH)
        total += float(cross_entropy(self.scores(contexts[part]), answers[part], reduction="sum"))
    return total / len(answers)

  def sample(self, *, seed: int, max_chars: int) -> str:
    """Write one character at a time until a boundary or max_chars."""
    generator = torch.Generator().manual_seed(seed)
    recent = [BOUNDARY] * self.config.context
    ids: list[int] = []
    while len(ids) < max_chars:
      context = torch.tensor([recent], device=self.config.device)
      with torch.no_grad():
        chances = torch.softmax(self.scores(context)[0], dim=0).cpu()
      chosen = int(torch.multinomial(chances, 1, generator=generator).item())
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
  generator = torch.Generator().manual_seed(config.seed)

  def weights(rows: int, columns: int, size: float) -> Tensor:
    values = torch.randn(rows, columns, generator=generator) * size
    return values.to(config.device).requires_grad_()

  def zeros(length: int) -> Tensor:
    return torch.zeros(length, device=config.device).requires_grad_()

  inputs = config.context * config.vector
  return NeuralModel(
    config=config,
    vocab=vocab,
    vectors=weights(vocab.size, config.vector, 1.0),
    hidden_weights=weights(inputs, config.hidden, 1.0 / math.sqrt(inputs)),
    hidden_bias=zeros(config.hidden),
    out_weights=weights(config.hidden, vocab.size, 0.01),
    out_bias=zeros(vocab.size),
  )


def train(
  texts: Sequence[str],
  vocab: Vocab,
  config: NeuralConfig,
  *,
  steps: int,
  validation: Sequence[str] = (),
) -> tuple[NeuralModel, list[tuple[int, float, float]]]:
  """Train for `steps`; history holds (step, training, validation) losses."""
  model = start(vocab, config)
  contexts, answers = examples(texts, vocab, config)
  generator = torch.Generator().manual_seed(config.seed)
  optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.rate, weight_decay=config.weight_decay
  )
  history: list[tuple[int, float, float]] = []
  for number in range(1, steps + 1):
    picked = torch.randint(len(answers), (config.batch,), generator=generator).to(config.device)
    mistake = cross_entropy(model.scores(contexts[picked]), answers[picked])
    slopes = torch.autograd.grad(mistake, model.parameters())
    for weight, slope in zip(model.parameters(), slopes, strict=True):
      weight.grad = slope
    optimizer.step()
    if validation and number % config.check_every == 0:
      history.append((number, model.loss(texts), model.loss(validation)))
  return model, history
