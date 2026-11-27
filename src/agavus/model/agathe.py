"""Agathe: attention over the whole context (docs/llm-lab.md §4, step 4).

Each token becomes a vector plus a position vector. Every layer lets each
position look back at all earlier ones (attention), then thinks about what it
gathered (a small network). The last vectors score the next token.
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, dropout, gelu, layer_norm

from agavus.model.bigram import BOUNDARY, Vocab
from agavus.model.data import Split
from agavus.model.training import IGNORED, History, Init, fit, mean_surprise, pick

EVAL_BATCH = 64
BYTE_IDS = 256
# Typical size of a starting weight.
SPREAD = 0.02
GROWTH = 4


@dataclass(frozen=True)
class AgatheConfig:
  """Size of the model and how it trains."""

  context: int = 64
  width: int = 128
  heads: int = 4
  layers: int = 4
  dropout: float = 0.2
  batch: int = 32
  rate: float = 3e-3
  weight_decay: float = 0.1
  check_every: int = 250
  seed: int = 0
  device: str = "cpu"


@dataclass(frozen=True)
class Layer:
  """One attention step and one thinking step."""

  query: Tensor
  key: Tensor
  value: Tensor
  mix: Tensor
  up: Tensor
  up_bias: Tensor
  down: Tensor
  down_bias: Tensor

  def parameters(self) -> list[Tensor]:
    """Everything that training adjusts in this layer."""
    return [
      self.query,
      self.key,
      self.value,
      self.mix,
      self.up,
      self.up_bias,
      self.down,
      self.down_bias,
    ]


def attend(layer: Layer, vectors: Tensor, heads: int) -> tuple[Tensor, Tensor]:
  """Each position gathers from earlier ones; also returns who looked at whom."""
  batch, length, width = vectors.shape
  size = width // heads

  def per_head(weights: Tensor) -> Tensor:
    return (vectors @ weights).view(batch, length, heads, size).transpose(1, 2)

  query, key, value = per_head(layer.query), per_head(layer.key), per_head(layer.value)
  scores = query @ key.transpose(-2, -1) / math.sqrt(size)
  future = torch.ones(length, length, dtype=torch.bool, device=vectors.device).triu(1)
  looks = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
  gathered = (looks @ value).transpose(1, 2).reshape(batch, length, width)
  return gathered @ layer.mix, looks


@dataclass(frozen=True)
class AgatheModel:
  """Token and position vectors, layers, and the output scores."""

  config: AgatheConfig
  vocab: Vocab
  tokens: Tensor
  positions: Tensor
  layers: tuple[Layer, ...]
  out_weights: Tensor
  out_bias: Tensor

  def parameters(self) -> list[Tensor]:
    """Everything that training adjusts."""
    inner = [weight for layer in self.layers for weight in layer.parameters()]
    return [self.tokens, self.positions, *inner, self.out_weights, self.out_bias]

  def size(self) -> int:
    """How many numbers training adjusts."""
    return sum(weight.numel() for weight in self.parameters())

  def run(self, ids: Tensor, *, training: bool = False) -> tuple[Tensor, list[Tensor]]:
    """Scores per position, and each layer's attention (who looked at whom)."""
    rate, width = self.config.dropout, self.config.width
    vectors = self.tokens[ids] + self.positions[: ids.shape[1]]
    all_looks: list[Tensor] = []
    for layer in self.layers:
      gathered, looks = attend(layer, layer_norm(vectors, (width,)), self.config.heads)
      all_looks.append(looks)
      vectors = vectors + dropout(gathered, rate, training)
      hidden = gelu(layer_norm(vectors, (width,)) @ layer.up + layer.up_bias)
      vectors = vectors + dropout(hidden @ layer.down + layer.down_bias, rate, training)
    return layer_norm(vectors, (width,)) @ self.out_weights + self.out_bias, all_looks

  def scores(self, ids: Tensor, *, training: bool = False) -> Tensor:
    """One score per next token at every position."""
    return self.run(ids, training=training)[0]

  def looks(self, text: str) -> list[Tensor]:
    """Attention per layer on a text: heads x positions x positions.

    Positions are the start boundary and then each token of the text (the
    last `context` of them), as the model sees them while writing.
    """
    ids = self.vocab.encode(text)[:-1][-self.config.context :]
    with torch.no_grad():
      _, looks = self.run(torch.tensor([ids], device=self.config.device))
    return [layer[0].cpu() for layer in looks]

  def loss(self, texts: Sequence[str]) -> float:
    """Mean surprise over every token, like the bigram's."""
    inputs, answers = windows(texts, self.vocab, self.config.context)
    return mean_surprise(
      lambda part: self.scores(part.to(self.config.device)), inputs, answers, EVAL_BATCH
    )

  def continuation(self, prefix: str, *, seed: int, max_chars: int) -> Iterator[int]:
    """Yield only new tokens, starting from the most recent window of the prefix."""
    generator = torch.Generator().manual_seed(seed)
    ids = self.vocab.encode(prefix)[:-1]
    for _ in range(max_chars):
      recent = torch.tensor([ids[-self.config.context :]], device=self.config.device)
      with torch.no_grad():
        chosen = pick(self.scores(recent)[0, -1], generator)
      if chosen == BOUNDARY:
        break
      ids.append(chosen)
      yield chosen

  def sample(self, *, seed: int, max_chars: int) -> str:
    """Decode a seeded continuation from the start boundary."""
    return self.vocab.decode(list(self.continuation("", seed=seed, max_chars=max_chars)))


def windows(texts: Sequence[str], vocab: Vocab, context: int) -> tuple[Tensor, Tensor]:
  """Every next token scored once, each with at least half a context behind it.

  Windows overlap by half. After the first, a window scores only its second
  half: the tokens at its start see little and were scored by the window before.
  """
  step = max(context // 2, 1)
  inputs: list[list[int]] = []
  answers: list[list[int]] = []
  for text in texts:
    ids = vocab.encode(text)
    start = 0
    while start == 0 or start + context - step < len(ids) - 1:
      targets = ids[start + 1 : start + context + 1]
      skip = 0 if start == 0 else context - step
      padding = context - len(targets)
      inputs.append(ids[start : start + len(targets)] + [BOUNDARY] * padding)
      answers.append([IGNORED] * skip + targets[skip:] + [IGNORED] * padding)
      start += step
  return torch.tensor(inputs), torch.tensor(answers)


def stream(texts: Sequence[str], vocab: Vocab) -> Tensor:
  """All training texts in one row of ids, joined by boundaries; one byte per id when it fits."""
  kind = torch.uint8 if vocab.size <= BYTE_IDS else torch.int64
  parts = [torch.tensor([BOUNDARY], dtype=kind)]
  parts.extend(torch.tensor(vocab.encode(text)[1:], dtype=kind) for text in texts)
  return torch.cat(parts)


def start(vocab: Vocab, config: AgatheConfig) -> AgatheModel:
  """Random small weights, reproducible by seed."""
  init = Init.seeded(config.seed, config.device)
  width, wide = config.width, config.width * GROWTH
  # Each residual path adds to the stream; keep their sum small at the start.
  path = SPREAD / math.sqrt(2 * config.layers)
  layers = tuple(
    Layer(
      query=init.matrix(width, width, SPREAD),
      key=init.matrix(width, width, SPREAD),
      value=init.matrix(width, width, SPREAD),
      mix=init.matrix(width, width, path),
      up=init.matrix(width, wide, SPREAD),
      up_bias=init.zeros(wide),
      down=init.matrix(wide, width, path),
      down_bias=init.zeros(width),
    )
    for _ in range(config.layers)
  )
  return AgatheModel(
    config=config,
    vocab=vocab,
    tokens=init.matrix(vocab.size, width, SPREAD),
    positions=init.matrix(config.context, width, SPREAD),
    layers=layers,
    out_weights=init.matrix(width, vocab.size, SPREAD),
    out_bias=init.zeros(vocab.size),
  )


def train(
  split: Split, vocab: Vocab, config: AgatheConfig, *, steps: int
) -> tuple[AgatheModel, History]:
  """Learn from split.train; history is checked on split.holdout, if any (see `fit`)."""
  model = start(vocab, config)
  ids = stream(split.train, vocab)
  offsets = torch.arange(config.context + 1)
  last_start = max(len(ids) - config.context - 1, 1)
  # Dropout draws from the global generator.
  torch.manual_seed(config.seed)

  def mistake(generator: torch.Generator) -> Tensor:
    starts = torch.randint(last_start, (config.batch, 1), generator=generator)
    rows = ids[(starts + offsets).clamp(max=len(ids) - 1)].long().to(config.device)
    scores = model.scores(rows[:, :-1], training=True)
    return cross_entropy(scores.flatten(0, 1), rows[:, 1:].flatten())

  return model, fit(model, mistake, config, steps=steps, watch=split)
