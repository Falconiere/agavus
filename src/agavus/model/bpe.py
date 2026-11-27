"""Byte-pair encoding: learn word pieces by gluing frequent pairs.

Starts from the 256 byte values, so every text can be encoded and decoded
exactly, including characters never seen (docs/llm-lab.md §4, step 2).
"""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise

from agavus.model.bigram import BOUNDARY, RESERVED

BYTES = 256
FIRST_MERGE = RESERVED + BYTES
MIN_COUNT = 2
# A word keeps the space before it; merges never cross words.
WORD = re.compile(r" ?\S+|\s+")
Pair = tuple[int, int]


def byte_ids(text: str) -> list[int]:
  """One token per byte, after the reserved ids."""
  return [value + RESERVED for value in text.encode("utf-8")]


def glue(ids: Sequence[int], pair: Pair, merged: int) -> list[int]:
  """Replace every occurrence of pair, left to right, by one token."""
  result: list[int] = []
  position = 0
  while position < len(ids):
    if tuple(ids[position : position + 2]) == pair:
      result.append(merged)
      position += 2
    else:
      result.append(ids[position])
      position += 1
  return result


@dataclass(frozen=True)
class BpeVocab:
  """Learned merges, in order; token ids follow from that order."""

  merges: tuple[Pair, ...]

  @property
  def size(self) -> int:
    """Number of tokens: reserved, bytes and merges."""
    return FIRST_MERGE + len(self.merges)

  @cached_property
  def _order(self) -> dict[Pair, int]:
    return {pair: FIRST_MERGE + rank for rank, pair in enumerate(self.merges)}

  @cached_property
  def _pieces(self) -> dict[int, bytes]:
    pieces = {value + RESERVED: bytes([value]) for value in range(BYTES)}
    for pair, token in self._order.items():
      pieces[token] = pieces[pair[0]] + pieces[pair[1]]
    return pieces

  def encode(self, text: str) -> list[int]:
    """Text to ids, wrapped in boundary tokens."""
    known: dict[str, list[int]] = {}
    ids = [BOUNDARY]
    for word in WORD.findall(text):
      if word not in known:
        known[word] = self._encode_word(word)
      ids += known[word]
    return [*ids, BOUNDARY]

  def decode(self, ids: Sequence[int]) -> str:
    """Ids to text; boundaries vanish, broken characters become visible marks."""
    joined = b"".join(self._pieces.get(token, b"") for token in ids)
    return joined.decode("utf-8", errors="replace")

  def piece(self, token: int) -> str:
    """Readable form of one token."""
    return self.decode([token])

  def _encode_word(self, word: str) -> list[int]:
    ids = byte_ids(word)
    while True:
      ranked = [(self._order[pair], pair) for pair in pairwise(ids) if pair in self._order]
      if not ranked:
        return ids
      merged, pair = min(ranked)
      ids = glue(ids, pair, merged)


def train_bpe(texts: Sequence[str], merges: int) -> BpeVocab:
  """Glue the most frequent pair, again and again; ties go to the lower ids."""
  words = Counter(word for text in texts for word in WORD.findall(text))
  split = {word: byte_ids(word) for word in words}
  learned: list[Pair] = []
  for number in range(merges):
    counts: Counter[Pair] = Counter()
    for word, ids in split.items():
      for pair in pairwise(ids):
        counts[pair] += words[word]
    if not counts:
      break
    best = min(counts, key=lambda pair: (-counts[pair], pair))
    if counts[best] < MIN_COUNT:
      break
    learned.append(best)
    split = {word: glue(ids, best, FIRST_MERGE + number) for word, ids in split.items()}
  return BpeVocab(tuple(learned))
