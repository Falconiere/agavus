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
from torch.nn.functional import (
  cross_entropy,
  dropout,
  gelu,
  layer_norm,
  scaled_dot_product_attention,
)

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
  bf16: bool = False


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


def attend(
  layer: Layer, vectors: Tensor, heads: int, *, show: bool = True
) -> tuple[Tensor, Tensor | None]:
  """Each position gathers from earlier ones; with `show`, also who looked at whom.

  Without `show`, PyTorch's fused attention never builds the positions x positions table,
  so memory grows with the context instead of its square.
  """
  query, key, value = _project(layer, vectors, heads)
  if not show:
    return _merge(layer, scaled_dot_product_attention(query, key, value, is_causal=True)), None
  length = vectors.shape[1]
  scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
  future = torch.ones(length, length, dtype=torch.bool, device=vectors.device).triu(1)
  looks = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
  return _merge(layer, looks @ value), looks


def attend_cached(
  layer: Layer, vectors: Tensor, heads: int, past: tuple[Tensor, Tensor] | None
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
  """Attention for positions after `past`, the keys and values earlier positions left.

  Without `past`, positions see each other causally. With it, one new position sees every
  earlier one. Returns the gathered vectors and the keys and values including the new ones.
  """
  query, key, value = _project(layer, vectors, heads)
  if past is not None:
    key, value = torch.cat((past[0], key), dim=2), torch.cat((past[1], value), dim=2)
  gathered = scaled_dot_product_attention(query, key, value, is_causal=past is None)
  return _merge(layer, gathered), (key, value)


def _project(layer: Layer, vectors: Tensor, heads: int) -> tuple[Tensor, Tensor, Tensor]:
  """Queries, keys and values, split per head: batch x heads x positions x size."""
  batch, length, width = vectors.shape

  def per_head(weights: Tensor) -> Tensor:
    return (vectors @ weights).view(batch, length, heads, width // heads).transpose(1, 2)

  return per_head(layer.query), per_head(layer.key), per_head(layer.value)


def _merge(layer: Layer, gathered: Tensor) -> Tensor:
  """Join the heads back into one vector per position and mix them."""
  batch, heads, length, size = gathered.shape
  return gathered.transpose(1, 2).reshape(batch, length, heads * size) @ layer.mix


def _think(layer: Layer, vectors: Tensor, rate: float, training: bool) -> Tensor:
  """The small network each position runs on what it gathered, added to its vector."""
  hidden = gelu(layer_norm(vectors, (vectors.shape[-1],)) @ layer.up + layer.up_bias)
  return vectors + dropout(hidden @ layer.down + layer.down_bias, rate, training)


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

  def run(
    self, ids: Tensor, *, training: bool = False, show: bool = False
  ) -> tuple[Tensor, list[Tensor]]:
    """Scores per position and, with `show`, each layer's attention (who looked at whom)."""
    rate, width = self.config.dropout, self.config.width
    vectors = self.tokens[ids] + self.positions[: ids.shape[1]]
    all_looks: list[Tensor] = []
    for layer in self.layers:
      gathered, looks = attend(layer, layer_norm(vectors, (width,)), self.config.heads, show=show)
      if looks is not None:
        all_looks.append(looks)
      vectors = _think(layer, vectors + dropout(gathered, rate, training), rate, training)
    return layer_norm(vectors, (width,)) @ self.out_weights + self.out_bias, all_looks

  def extend(self, ids: Sequence[int], cache: list[tuple[Tensor, Tensor]]) -> Tensor:
    """Next-token scores after `ids`, placed after the positions `cache` holds; updates it.

    An empty cache starts at position 0. With a cache, pass one new id at a time.
    """
    width = self.config.width
    offset = cache[0][0].shape[2] if cache else 0
    tensor = torch.tensor([list(ids)], device=self.config.device)
    vectors = self.tokens[tensor] + self.positions[offset : offset + len(ids)]
    updated: list[tuple[Tensor, Tensor]] = []
    for index, layer in enumerate(self.layers):
      past = cache[index] if cache else None
      gathered, keys = attend_cached(layer, layer_norm(vectors, (width,)), self.config.heads, past)
      updated.append(keys)
      vectors = _think(layer, vectors + gathered, 0.0, False)
    cache[:] = updated
    return (layer_norm(vectors[0, -1], (width,)) @ self.out_weights + self.out_bias).detach()

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
      _, looks = self.run(torch.tensor([ids], device=self.config.device), show=True)
    return [layer[0].cpu() for layer in looks]

  def loss(self, texts: Sequence[str]) -> float:
    """Mean surprise over every token, like the bigram's."""
    inputs, answers = windows(texts, self.vocab, self.config.context)
    return mean_surprise(
      lambda part: self.scores(part.to(self.config.device)), inputs, answers, EVAL_BATCH
    )

  def continuation(self, prefix: str, *, seed: int, max_chars: int) -> Iterator[int]:
    """Yield only new tokens, starting from the most recent window of the prefix.

    Keys and values of earlier positions are cached while the window has room; once it is
    full, the window slides and is read again, as without a cache.
    """
    generator = torch.Generator().manual_seed(seed)
    context = self.config.context
    ids = self.vocab.encode(prefix)[:-1]
    cache: list[tuple[Tensor, Tensor]] = []
    with torch.no_grad():
      scores = self.extend(ids[-context:], cache)
    for _ in range(max_chars):
      chosen = pick(scores, generator)
      if chosen == BOUNDARY:
        break
      ids.append(chosen)
      yield chosen
      with torch.no_grad():
        if cache[0][0].shape[2] < context:
          scores = self.extend([chosen], cache)
        else:
          cache.clear()
          scores = self.extend(ids[-context:], cache)

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
    kind = torch.device(config.device).type
    with torch.autocast(kind, dtype=torch.bfloat16, enabled=config.bf16):
      scores = model.scores(rows[:, :-1], training=True)
    return cross_entropy(scores.float().flatten(0, 1), rows[:, 1:].flatten())

  return model, fit(model, mistake, config, steps=steps, watch=split)
