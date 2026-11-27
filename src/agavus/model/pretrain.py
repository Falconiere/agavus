"""Agathe's pretraining data (docs/llm-lab.md §9.3): sources, cleaning, tokenized shards.

Each source is read lazily, cleaned (empty texts, exact duplicates across all sources,
texts sharing a run with an exam statute, and, where asked, texts the personal-data scan
flags), tokenized once and written as `uint16` shards: every document followed by the end
token. A manifest records counts, licenses and each shard's SHA-256, so a rented machine can
verify its copy.
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from agavus.corpus.anonymize import contains_personal_data
from agavus.model.extra import without_leaks
from agavus.model.federal import save_bytes
from agavus.model.tokenizer import END

PRETRAIN_DIR = Path("lab") / "pretrain"
SHARD_TOKENS = 100_000_000
CHUNK = 10_000


@dataclass(frozen=True)
class Source:
  """A named stream of texts, its license, and whether to run the personal-data scan."""

  name: str
  license: str
  texts: Callable[[], Iterable[str]]
  scan: bool = False


@dataclass
class Pipeline:
  """What every source shares: the tokenizer, the exam guard, seen texts and the output."""

  tokenizer: Tokenizer
  statutes: Sequence[str]
  out: Path
  shard_tokens: int = SHARD_TOKENS
  seen: set[bytes] = field(default_factory=set)


@dataclass
class Shards:
  """Counts for one source while its shards are written."""

  docs: int = 0
  chars: int = 0
  tokens: int = 0
  dropped: dict[str, int] = field(default_factory=lambda: dict.fromkeys(("dup", "leak", "pii"), 0))
  files: list[dict[str, object]] = field(default_factory=list)


def parquet_texts(paths: Sequence[Path], column: str = "text") -> Iterator[str]:
  """The `column` of every row, file by file, without loading a whole file at once."""
  for path in paths:
    for batch in pq.ParquetFile(path).iter_batches(columns=[column], batch_size=CHUNK):
      yield from (str(value) for value in batch.column(0).to_pylist() if value)


def _key(text: str) -> bytes:
  return hashlib.blake2b(" ".join(text.split()).encode(), digest_size=16).digest()


def clean(
  source: Source, statutes: Sequence[str], seen: set[bytes], counts: Shards
) -> Iterator[list[str]]:
  """Chunks of texts that survive deduplication, the leak check and, if asked, the scan."""
  chunk: list[str] = []
  for raw in source.texts():
    text = raw.strip()
    if not text:
      continue
    key = _key(text)
    if key in seen:
      counts.dropped["dup"] += 1
      continue
    seen.add(key)
    if source.scan and contains_personal_data(text):
      counts.dropped["pii"] += 1
      continue
    chunk.append(text)
    if len(chunk) == CHUNK:
      yield _without_leaks(chunk, statutes, counts)
      chunk = []
  if chunk:
    yield _without_leaks(chunk, statutes, counts)


def _without_leaks(chunk: list[str], statutes: Sequence[str], counts: Shards) -> list[str]:
  kept = list(without_leaks(chunk, statutes))
  counts.dropped["leak"] += len(chunk) - len(kept)
  return kept


def write_source(source: Source, pipeline: Pipeline) -> Shards:
  """Clean, tokenize and shard one source under `out/<name>/`; returns its counts."""
  tokenizer, shard_tokens = pipeline.tokenizer, pipeline.shard_tokens
  end = tokenizer.token_to_id(END)
  if end is None:
    raise ValueError(f"the tokenizer has no {END} token")
  counts = Shards()
  pending: list[np.ndarray] = []
  size = 0
  for chunk in clean(source, pipeline.statutes, pipeline.seen, counts):
    for encoding in tokenizer.encode_batch(chunk, add_special_tokens=False):
      pending.append(np.array([*encoding.ids, end], dtype=np.uint16))
      size += len(encoding.ids) + 1
    counts.docs += len(chunk)
    counts.chars += sum(map(len, chunk))
    while size >= shard_tokens:
      size = _flush(pending, shard_tokens, pipeline.out / source.name, counts)
  if size:
    _flush(pending, size, pipeline.out / source.name, counts)
  return counts


def _flush(pending: list[np.ndarray], tokens: int, folder: Path, counts: Shards) -> int:
  """Write the first `tokens` pending ids as one shard; keep the rest pending."""
  joined = np.concatenate(pending)
  shard, rest = joined[:tokens], joined[tokens:]
  path = folder / f"shard-{len(counts.files):05d}.bin"
  payload = shard.tobytes()
  save_bytes(path, payload)
  counts.files.append(
    {"file": path.name, "tokens": int(shard.size), "sha256": hashlib.sha256(payload).hexdigest()}
  )
  counts.tokens += int(shard.size)
  pending[:] = [rest] if rest.size else []
  return int(rest.size)


def write_manifest(
  out: Path, tokenizer_sha256: str, sources: dict[str, tuple[str, Shards]]
) -> Path:
  """`out/manifest.json`: tokenizer hash, then per source its license, counts and shards."""
  manifest = {
    "tokenizer_sha256": tokenizer_sha256,
    "sources": {
      name: {
        "license": license,
        "docs": counts.docs,
        "chars": counts.chars,
        "tokens": counts.tokens,
        "dropped": counts.dropped,
        "shards": counts.files,
      }
      for name, (license, counts) in sources.items()
    },
  }
  path = out / "manifest.json"
  save_bytes(path, json.dumps(manifest, indent=2, ensure_ascii=False).encode())
  return path


def read_shard(path: Path) -> np.ndarray:
  """A shard's token ids, mapped from disk rather than read into memory."""
  return np.memmap(path, dtype=np.uint16, mode="r")
