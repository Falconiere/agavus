"""Pretraining shards from synthetic sources: cleaning, sharding, manifest, Parquet reading."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from agavus.model.extra import LEAK_RUN
from agavus.model.pretrain import (
  Pipeline,
  Source,
  parquet_texts,
  read_shard,
  write_manifest,
  write_source,
)
from agavus.model.tokenizer import END, decode, train_tokenizer

EXAM = "Art. 22. " + "medida protetiva de urgência " * 8
LAWS = [f"Art. {n}º Esta Lei dispõe sobre a medida número {n}." for n in range(1, 30)]
CASES = [
  "Decisão: concedo a medida protetiva.",
  "Decisão: concedo a medida protetiva.",
  "Contato do advogado: fulano@exemplo.com.br",
  "Citação longa: " + EXAM[: LEAK_RUN + 10],
  "Decisão: indefiro o pedido.",
]


def _tokenizer_and_sources() -> tuple[list[Source], Tokenizer]:
  tokenizer = train_tokenizer([*LAWS, *CASES, EXAM] * 3, 300)
  laws = Source("planalto", "public domain", lambda: [*LAWS, "  ", CASES[0]])
  cases = Source("djen", "anonymized", lambda: CASES, scan=True)
  return [laws, cases], tokenizer


def _documents(folder: Path, end: int) -> list[list[int]]:
  ids = np.concatenate([read_shard(path) for path in sorted(folder.glob("shard-*.bin"))])
  documents: list[list[int]] = []
  current: list[int] = []
  for token in ids.tolist():
    if token == end:
      documents.append(current)
      current = []
    else:
      current.append(token)
  assert not current
  return documents


def test_sources_are_cleaned_sharded_and_decode_back(tmp_path: Path) -> None:
  (laws, cases), tokenizer = _tokenizer_and_sources()
  pipeline = Pipeline(tokenizer, [EXAM], tmp_path, shard_tokens=64)
  first = write_source(laws, pipeline)
  second = write_source(cases, pipeline)
  assert (first.docs, first.dropped) == (len(LAWS) + 1, {"dup": 0, "leak": 0, "pii": 0})
  assert second.docs == 1
  assert second.dropped == {"dup": 2, "leak": 1, "pii": 1}
  assert len(first.files) > 1
  end = 0
  assert tokenizer.token_to_id(END) == end
  texts = [decode(tokenizer, ids) for ids in _documents(tmp_path / "planalto", end)]
  assert texts == [*LAWS, CASES[0]]
  assert [decode(tokenizer, ids) for ids in _documents(tmp_path / "djen", end)] == [CASES[4]]
  manifest = json.loads(
    write_manifest(tmp_path, "abc", {"planalto": ("public domain", first)}).read_text()
  )
  shards = manifest["sources"]["planalto"]["shards"]
  for shard in shards:
    payload = (tmp_path / "planalto" / shard["file"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == shard["sha256"]
  assert sum(shard["tokens"] for shard in shards) == first.tokens


def test_parquet_texts_reads_the_column_and_skips_empty_rows(tmp_path: Path) -> None:
  path = tmp_path / "wiki.parquet"
  pq.write_table(pa.table({"title": ["a", "b", "c"], "text": ["Um.", "", "Dois."]}), path)
  assert list(parquet_texts([path, path])) == ["Um.", "Dois.", "Um.", "Dois."]
