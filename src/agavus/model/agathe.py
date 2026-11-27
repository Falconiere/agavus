"""Agathe: a Qwen-style decoder over BPE tokens (docs/llm-lab.md §9.2).

Each token becomes a vector. Every layer lets each position look back at earlier ones
(attention), then thinks about what it gathered (a gated network). Positions turn queries
and keys (RoPE) instead of adding a learned table, RMSNorm keeps vectors in scale, several
query heads share each key/value head (GQA), and the token table also scores the next token
(tied embeddings). No biases anywhere.
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.nn.functional import (
  cross_entropy,
  dropout,
  rms_norm,
  scaled_dot_product_attention,
  silu,
)

from agavus.model.data import Split
from agavus.model.tokenizer import END, SPECIAL, decode, encode
from agavus.model.training import IGNORED, History, Init, fit, mean_surprise, pick

EVAL_BATCH = 64
# Typical size of a starting weight.
SPREAD = 0.02
EPSILON = 1e-6
# SwiGLU's hidden size keeps the parameter count of a 4x GELU network, rounded for kernels.
HIDDEN_RATIO = 8 / 3
HIDDEN_STEP = 64
# The end token separates documents and stops writing; special tokens take the first ids.
END_ID = SPECIAL.index(END)
# Training streams store ids in two bytes.
STREAM_IDS = 1 << 16
Cache = list[tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class AgatheConfig:
  """Size of the model and how it trains."""

  context: int = 1024
  width: int = 512
  heads: int = 8
  kv_heads: int = 2
  layers: int = 8
  rope_base: float = 10000.0
  dropout: float = 0.0
  batch: int = 16
  rate: float = 3e-4
  weight_decay: float = 0.1
  check_every: int = 500
  seed: int = 0
  device: str = "cpu"
  bf16: bool = False


def check_config(config: AgatheConfig) -> None:
  """Refuse sizes the heads cannot split evenly or RoPE cannot pair up."""
  sizes = (config.context, config.width, config.heads, config.kv_heads, config.layers)
  if min(sizes) < 1:
    raise ValueError("context, width, heads, kv_heads and layers must be positive")
  if config.width % config.heads:
    raise ValueError(f"width {config.width} must divide into {config.heads} heads")
  if config.heads % config.kv_heads:
    raise ValueError(f"{config.heads} heads must share {config.kv_heads} key/value heads")
  if (config.width // config.heads) % 2:
    raise ValueError("head size (width / heads) must be even for RoPE")


def hidden_size(width: int) -> int:
  """SwiGLU hidden width: the multiple of 64 nearest 8/3 of the width."""
  return HIDDEN_STEP * max(1, round(width * HIDDEN_RATIO / HIDDEN_STEP))


def framed(tokenizer: Tokenizer, text: str) -> list[int]:
  """Ids for `text` between two end tokens."""
  return [END_ID, *encode(tokenizer, text), END_ID]


@dataclass(frozen=True)
class Layer:
  """One attention step and one gated thinking step, each after its own RMSNorm."""

  attn_gain: Tensor
  query: Tensor
  key: Tensor
  value: Tensor
  mix: Tensor
  mlp_gain: Tensor
  gate: Tensor
  up: Tensor
  down: Tensor

  def parameters(self) -> list[Tensor]:
    """Everything that training adjusts in this layer."""
    return [
      self.attn_gain,
      self.query,
      self.key,
      self.value,
      self.mix,
      self.mlp_gain,
      self.gate,
      self.up,
      self.down,
    ]

  def attend(
    self,
    vectors: Tensor,
    config: AgatheConfig,
    positions: Tensor,
    past: tuple[Tensor, Tensor] | None = None,
    looks: list[Tensor] | None = None,
  ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Attention for `positions`, after the keys and values in `past`, if any.

    Keeps at most `context` keys and values: the oldest drop out, which RoPE allows because
    only the distance between a query and a key matters. With a `looks` list (and no past),
    also appends who looked at whom: batch x heads x positions x positions.
    """
    batch, length, width = vectors.shape
    size = width // config.heads

    def per_head(weights: Tensor, count: int) -> Tensor:
      return (vectors @ weights).view(batch, length, count, size).transpose(1, 2)

    cos, sin = rotary(size, positions, config.rope_base)
    query = rotate(per_head(self.query, config.heads), cos, sin)
    keys = rotate(per_head(self.key, config.kv_heads), cos, sin)
    values = per_head(self.value, config.kv_heads)
    if past is not None:
      keys = torch.cat((past[0], keys), dim=2)[:, :, -config.context :]
      values = torch.cat((past[1], values), dim=2)[:, :, -config.context :]
    group = config.heads // config.kv_heads
    shared = keys.repeat_interleave(group, dim=1)
    gathered = scaled_dot_product_attention(
      query, shared, values.repeat_interleave(group, dim=1), is_causal=past is None
    )
    if looks is not None:
      scores = query @ shared.transpose(-2, -1) / math.sqrt(size)
      future = torch.ones(length, length, dtype=torch.bool, device=vectors.device).triu(1)
      looks.append(torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1))
    merged = gathered.transpose(1, 2).reshape(batch, length, width) @ self.mix
    return merged, (keys, values)


def rotary(size: int, positions: Tensor, base: float) -> tuple[Tensor, Tensor]:
  """Cosines and sines turning each pair of a head at each position: positions x size."""
  steps = torch.arange(0, size, 2, dtype=torch.float32, device=positions.device)
  angles = positions.float()[:, None] * base ** (-steps / size)[None, :]
  angles = torch.cat((angles, angles), dim=-1)
  return angles.cos(), angles.sin()


def rotate(heads: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
  """Turn the two halves of each head together (rotate-half convention)."""
  half = heads.shape[-1] // 2
  turned = torch.cat((-heads[..., half:], heads[..., :half]), dim=-1)
  return heads * cos + turned * sin


def _swiglu(layer: Layer, vectors: Tensor) -> Tensor:
  """The gated network: silu(x Wg) * (x Wu), brought back to the width."""
  return (silu(vectors @ layer.gate) * (vectors @ layer.up)) @ layer.down


@dataclass(frozen=True)
class AgatheModel:
  """Tokenizer, token table (also the output), layers, and a final RMSNorm gain."""

  config: AgatheConfig
  tokenizer: Tokenizer
  tokens: Tensor
  layers: tuple[Layer, ...]
  gain: Tensor

  def parameters(self) -> list[Tensor]:
    """Everything that training adjusts."""
    inner = [weight for layer in self.layers for weight in layer.parameters()]
    return [self.tokens, *inner, self.gain]

  def size(self) -> int:
    """How many numbers training adjusts."""
    return sum(weight.numel() for weight in self.parameters())

  def run(
    self,
    ids: Tensor,
    offset: int = 0,
    cache: Cache | None = None,
    training: bool = False,
    looks: list[Tensor] | None = None,
  ) -> Tensor:
    """Scores at every position of `ids`, which start at absolute position `offset`.

    With a `cache` list, earlier keys and values are read from it and it is updated.
    """
    width, rate = (self.config.width,), self.config.dropout
    positions = torch.arange(offset, offset + ids.shape[1], device=ids.device)
    vectors = self.tokens[ids]
    updated: Cache = []
    for index, layer in enumerate(self.layers):
      past = cache[index] if cache else None
      normed = rms_norm(vectors, width, layer.attn_gain, EPSILON)
      gathered, kept = layer.attend(normed, self.config, positions, past, looks)
      updated.append(kept)
      vectors = vectors + dropout(gathered, rate, training)
      thought = _swiglu(layer, rms_norm(vectors, width, layer.mlp_gain, EPSILON))
      vectors = vectors + dropout(thought, rate, training)
    if cache is not None:
      cache[:] = updated
    return rms_norm(vectors, width, self.gain, EPSILON) @ self.tokens.T

  def scores(self, ids: Tensor, *, training: bool = False) -> Tensor:
    """One score per next token at every position 0..T-1: batch x positions x vocab."""
    return self.run(ids, training=training)

  def extend(self, ids: Sequence[int], offset: int, cache: Cache) -> Tensor:
    """Next-token scores after `ids`, placed at `offset` after what `cache` holds."""
    with torch.no_grad():
      found = self.run(torch.tensor([list(ids)], device=self.config.device), offset, cache)
    return found[0, -1]

  def looks(self, text: str) -> list[Tensor]:
    """Attention per layer on a text: heads x positions x positions.

    Positions are the start token and then each token of the text (the last `context`
    of them), as the model sees them while writing.
    """
    ids = framed(self.tokenizer, text)[:-1][-self.config.context :]
    found: list[Tensor] = []
    with torch.no_grad():
      self.run(torch.tensor([ids], device=self.config.device), looks=found)
    return [layer[0].cpu() for layer in found]

  def surprise(self, texts: Sequence[str]) -> tuple[float, int]:
    """Mean nats per scored token over `texts`, and how many tokens were scored."""
    inputs, answers = windows(texts, self.tokenizer, self.config.context)
    device = self.config.device
    mean = mean_surprise(lambda rows: self.scores(rows.to(device)), inputs, answers, EVAL_BATCH)
    return mean, int((answers != IGNORED).sum())

  def loss(self, texts: Sequence[str]) -> float:
    """Mean nats per token; every token of every text scored once."""
    return self.surprise(texts)[0]

  def bits_per_char(self, texts: Sequence[str]) -> float:
    """Total bits over every scored token per character: comparable across tokenizers."""
    mean, scored = self.surprise(texts)
    return mean * scored / (math.log(2) * max(1, sum(map(len, texts))))

  def continuation(self, prefix: str, *, seed: int, max_tokens: int) -> Iterator[int]:
    """Yield new token ids after the last `context` tokens of the prefix, until the end id.

    Keys and values stay cached; past `context`, the oldest drop out instead of the window
    being read again.
    """
    generator = torch.Generator().manual_seed(seed)
    window = framed(self.tokenizer, prefix)[:-1][-self.config.context :]
    cache: Cache = []
    found = self.extend(window, 0, cache)
    for position in range(len(window), len(window) + max_tokens):
      chosen = pick(found, generator)
      if chosen == END_ID:
        return
      yield chosen
      found = self.extend([chosen], position, cache)

  def sample(self, *, seed: int, max_chars: int) -> str:
    """Decode a seeded continuation from the start token, at most `max_chars` long."""
    written = self.continuation("", seed=seed, max_tokens=max_chars)
    return decode(self.tokenizer, list(written))[:max_chars]


def windows(texts: Sequence[str], tokenizer: Tokenizer, context: int) -> tuple[Tensor, Tensor]:
  """Every next token scored once, each with at least half a context behind it.

  Windows overlap by half. After the first, a window scores only its second
  half: the tokens at its start see little and were scored by the window before.
  """
  step = max(context // 2, 1)
  inputs: list[list[int]] = []
  answers: list[list[int]] = []
  for text in texts:
    ids = framed(tokenizer, text)
    start = 0
    while start == 0 or start + context - step < len(ids) - 1:
      targets = ids[start + 1 : start + context + 1]
      skip = 0 if start == 0 else context - step
      padding = context - len(targets)
      inputs.append(ids[start : start + len(targets)] + [END_ID] * padding)
      answers.append([IGNORED] * skip + targets[skip:] + [IGNORED] * padding)
      start += step
  return torch.tensor(inputs), torch.tensor(answers)


def stream(texts: Sequence[str], tokenizer: Tokenizer) -> Tensor:
  """All training texts in one row of ids joined by end ids, two bytes per id."""
  if tokenizer.get_vocab_size() > STREAM_IDS:
    raise ValueError(f"ids are stored in two bytes; the vocabulary must stay ≤ {STREAM_IDS}")
  parts = [torch.tensor([END_ID], dtype=torch.uint16)]
  parts.extend(torch.tensor(framed(tokenizer, text)[1:], dtype=torch.uint16) for text in texts)
  return torch.cat(parts)


def start(tokenizer: Tokenizer, config: AgatheConfig) -> AgatheModel:
  """Random small weights, reproducible by seed; RMSNorm gains start at one."""
  check_config(config)
  if tokenizer.token_to_id(END) != END_ID:
    raise ValueError(f"{END} must have id {END_ID}")
  init = Init.seeded(config.seed, config.device)
  width, hidden = config.width, hidden_size(config.width)
  shared = config.kv_heads * (width // config.heads)
  # Each residual path adds to the stream; keep their sum small at the start.
  path = SPREAD / math.sqrt(2 * config.layers)

  def ones() -> Tensor:
    return torch.ones(width, device=config.device).requires_grad_()

  layers = tuple(
    Layer(
      attn_gain=ones(),
      query=init.matrix(width, width, SPREAD),
      key=init.matrix(width, shared, SPREAD),
      value=init.matrix(width, shared, SPREAD),
      mix=init.matrix(width, width, path),
      mlp_gain=ones(),
      gate=init.matrix(width, hidden, SPREAD),
      up=init.matrix(width, hidden, SPREAD),
      down=init.matrix(hidden, width, path),
    )
    for _ in range(config.layers)
  )
  tokens = init.matrix(tokenizer.get_vocab_size(), width, SPREAD)
  return AgatheModel(config, tokenizer, tokens, layers, ones())


def train(
  split: Split, tokenizer: Tokenizer, config: AgatheConfig, *, steps: int
) -> tuple[AgatheModel, History]:
  """Learn from split.train; history is checked on split.holdout, if any (see `fit`)."""
  model = start(tokenizer, config)
  ids = stream(split.train, tokenizer)
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
