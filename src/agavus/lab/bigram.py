"""Character bigram: the smallest language model (docs/llm-lab.md §4).

P(next | current) = (count + k) / (total + k * V). Training is counting;
the loss is the mean of -ln P(true next character).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BOUNDARY = 0
UNKNOWN = 1
RESERVED = 2
SMOOTHING = 1.0


@dataclass(frozen=True)
class CharVocab:
  """Characters known from training; ids 0 and 1 are boundary and unknown."""

  chars: tuple[str, ...]

  @property
  def size(self) -> int:
    """Number of tokens, reserved ones included (V)."""
    return len(self.chars) + RESERVED

  def encode(self, text: str) -> list[int]:
    """Text to ids, wrapped in boundary tokens."""
    index = {char: position + RESERVED for position, char in enumerate(self.chars)}
    return [BOUNDARY, *(index.get(char, UNKNOWN) for char in text), BOUNDARY]

  def decode(self, ids: Sequence[int]) -> str:
    """Ids to text; boundaries vanish, unknown becomes a visible mark."""
    pieces: list[str] = []
    for token in ids:
      if token == UNKNOWN:
        pieces.append("\N{REPLACEMENT CHARACTER}")
      elif token != BOUNDARY:
        pieces.append(self.chars[token - RESERVED])
    return "".join(pieces)


@dataclass(frozen=True)
class BigramModel:
  """One probability row per current token; every row sums to 1."""

  vocab: CharVocab
  probs: NDArray[np.float64]

  def probability(self, current: str, following: str) -> float:
    """P(following | current) for two single characters."""
    row = self.vocab.encode(current)[1]
    column = self.vocab.encode(following)[1]
    return float(self.probs[row, column])

  def loss(self, texts: Sequence[str]) -> float:
    """Mean -ln P of each true next token; lower is better."""
    counts = count_pairs(texts, self.vocab)
    return float(-(counts * np.log(self.probs)).sum() / counts.sum())

  def sample(self, *, seed: int, max_chars: int) -> str:
    """Draw one token at a time until a boundary or max_chars."""
    rng = np.random.default_rng(seed)
    ids: list[int] = []
    current = BOUNDARY
    while len(ids) < max_chars:
      current = int(rng.choice(self.vocab.size, p=self.probs[current]))
      if current == BOUNDARY:
        break
      ids.append(current)
    return self.vocab.decode(ids)


def build_vocab(texts: Sequence[str]) -> CharVocab:
  """Sorted distinct characters of the training texts."""
  return CharVocab(chars=tuple(sorted(set().union(*texts))))


def count_pairs(texts: Sequence[str], vocab: CharVocab) -> NDArray[np.float64]:
  """counts[current, next] over all texts; pairs never cross two texts."""
  counts = np.zeros((vocab.size, vocab.size), dtype=np.float64)
  for text in texts:
    ids = np.array(vocab.encode(text))
    np.add.at(counts, (ids[:-1], ids[1:]), 1.0)
  return counts


def train_bigram(texts: Sequence[str], *, smoothing: float = SMOOTHING) -> BigramModel:
  """Count, add smoothing, normalize each row."""
  if not texts:
    raise ValueError("need at least one training text")
  if smoothing <= 0:
    raise ValueError("smoothing must be positive: unseen pairs need P > 0")
  vocab = build_vocab(texts)
  counts = count_pairs(texts, vocab) + smoothing
  return BigramModel(vocab=vocab, probs=counts / counts.sum(axis=1, keepdims=True))


def uniform_loss(vocab: CharVocab) -> float:
  """Loss of guessing every token equally: ln V, the floor to beat."""
  return math.log(vocab.size)
