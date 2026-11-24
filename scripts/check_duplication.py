"""Detect duplicated code: any shared run of >= MIN_TOKENS tokens fails the build.

Python-native replacement for jscpd. Comments and whitespace are ignored.
"""

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from tokenize import TokenInfo

from pyutils import code_tokens, iter_python_files

MIN_TOKENS = 30
WindowKey = tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class Location:
  """A token position inside one file."""

  file_idx: int
  start: int


@dataclass(frozen=True)
class Clone:
  """A duplicated token run shared by two locations."""

  left: Location
  right: Location
  length: int


def _key(token: TokenInfo) -> tuple[int, str]:
  return (token.type, token.string)


def _build_index(files: Sequence[Sequence[TokenInfo]]) -> dict[WindowKey, list[Location]]:
  index: dict[WindowKey, list[Location]] = defaultdict(list)
  for file_idx, tokens in enumerate(files):
    for start in range(len(tokens) - MIN_TOKENS + 1):
      key = tuple(_key(t) for t in tokens[start : start + MIN_TOKENS])
      index[key].append(Location(file_idx, start))
  return index


def _run_length(left: Sequence[TokenInfo], right: Sequence[TokenInfo], a: int, b: int) -> int:
  length = 0
  while a + length < len(left) and b + length < len(right):
    if _key(left[a + length]) != _key(right[b + length]):
      break
    length += 1
  return length


def _contains(outer: Clone, inner: Clone) -> bool:
  if (outer.left.file_idx, outer.right.file_idx) != (inner.left.file_idx, inner.right.file_idx):
    return False
  return (
    outer.left.start <= inner.left.start
    and outer.right.start <= inner.right.start
    and inner.left.start + inner.length <= outer.left.start + outer.length
    and inner.right.start + inner.length <= outer.right.start + outer.length
  )


def find_clones(files: Sequence[Sequence[TokenInfo]]) -> list[Clone]:
  """Return maximal duplicated token runs across (and within) the given files."""
  clones: list[Clone] = []
  for locations in _build_index(files).values():
    for left, right in combinations(locations, 2):
      length = _run_length(files[left.file_idx], files[right.file_idx], left.start, right.start)
      clones.append(Clone(left, right, length))
  maximal: list[Clone] = []
  for clone in sorted(clones, key=lambda c: -c.length):
    if not any(_contains(kept, clone) for kept in maximal):
      maximal.append(clone)
  return maximal


def describe(paths: Sequence[Path], tokens: Sequence[Sequence[TokenInfo]], clone: Clone) -> str:
  """Format a clone as a human-readable violation message."""

  def where(loc: Location) -> str:
    file_tokens = tokens[loc.file_idx]
    first = file_tokens[loc.start].start[0]
    last = file_tokens[loc.start + clone.length - 1].end[0]
    return f"{paths[loc.file_idx]}:{first}-{last}"

  return f"duplicate code ({clone.length} tokens): {where(clone.left)} <-> {where(clone.right)}"


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point; returns 1 when any duplication is found."""
  parser = argparse.ArgumentParser(description="Detect duplicated code.")
  parser.add_argument("roots", nargs="*", default=["src", "scripts"])
  args = parser.parse_args(argv)
  paths = list(iter_python_files(args.roots))
  tokens = [code_tokens(path.read_text(encoding="utf-8")) for path in paths]
  clones = find_clones(tokens)
  for clone in clones:
    print(describe(paths, tokens, clone))
  return 1 if clones else 0


if __name__ == "__main__":
  raise SystemExit(main())
