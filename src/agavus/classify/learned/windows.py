"""Raw tokenizer windows: complete documents, indivisible statement pairs."""

from collections.abc import Sequence

from tokenizers import Tokenizer

WINDOW_TOKENS = 480
OVERLAP = 64
MAX_TOKENS = 512


def _special(tokenizer: Tokenizer, name: str, fallback: str) -> int:
  value = tokenizer.token_to_id(name)
  if value is None:
    value = tokenizer.token_to_id(fallback)
  if value is None:
    raise ValueError(f"tokenizer lacks boundary token {name}")
  return value


def _wrapped(tokenizer: Tokenizer, ids: Sequence[int]) -> tuple[int, ...]:
  result = (_special(tokenizer, "<s>", "[CLS]"), *ids, _special(tokenizer, "</s>", "[SEP]"))
  if len(result) > MAX_TOKENS:
    raise ValueError("prefix and boundaries exceed encoder capacity")
  return result


def document_windows(tokenizer: Tokenizer, text: str) -> tuple[tuple[int, ...], ...]:
  """Every payload token is covered; prefix and boundaries repeat in every window."""
  ids = tokenizer.encode(text, add_special_tokens=False).ids
  prefix = tokenizer.encode("passage: ", add_special_tokens=False).ids
  windows: list[tuple[int, ...]] = []
  for start in range(0, max(1, len(ids)), WINDOW_TOKENS - OVERLAP):
    windows.append(_wrapped(tokenizer, [*prefix, *ids[start : start + WINDOW_TOKENS]]))
    if start + WINDOW_TOKENS >= len(ids):
      break
  return tuple(windows)


def relation_window(tokenizer: Tokenizer, passage: str, statement: str) -> tuple[int, ...]:
  """Preserve the whole pair; truncating its ending could reverse its label."""
  ids = tokenizer.encode(f"query: {statement} passage: {passage}", add_special_tokens=False).ids
  if len(ids) > WINDOW_TOKENS:
    raise ValueError("oversized statement pair; no truncation permitted")
  return _wrapped(tokenizer, ids)
