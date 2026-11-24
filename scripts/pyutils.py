"""Shared source-analysis helpers for repo quality scripts."""

import io
import tokenize
from collections.abc import Iterator, Sequence
from pathlib import Path

_IGNORED_TOKENS = frozenset(
  {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
    tokenize.ENCODING,
  }
)


def code_tokens(source: str) -> list[tokenize.TokenInfo]:
  """Return significant tokens only (no comments, blanks, or layout tokens)."""
  tokens = tokenize.generate_tokens(io.StringIO(source).readline)
  return [token for token in tokens if token.type not in _IGNORED_TOKENS]


def code_lines(source: str) -> set[int]:
  """Return 1-based line numbers containing real code (no comments or blanks)."""
  return {token.start[0] for token in code_tokens(source)}


def iter_python_files(roots: Sequence[str]) -> Iterator[Path]:
  """Yield every Python file under the given roots, in stable order."""
  for root in roots:
    yield from sorted(Path(root).rglob("*.py"))
