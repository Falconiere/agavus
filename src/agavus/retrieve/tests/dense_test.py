"""Ranking by meaning with a synthetic encoder (no weights, no network)."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from agavus.corpus.httpfixtures import sample_text
from agavus.corpus.storage import CorpusStorage
from agavus.retrieve.dense import CHUNK_WORDS, chunk, index_statutes, search_statutes
from agavus.retrieve.encoder import MODEL_VERSION, FloatArray, load_encoder


class FakeEncoder:
  """Maps money-related texts to one axis and everything else to another."""

  def encode_queries(self, texts: Sequence[str]) -> FloatArray:
    """Axis by keyword."""
    return np.array([[1.0, 0.0] if "pensão" in text else [0.0, 1.0] for text in texts])

  def encode_passages(self, texts: Sequence[str]) -> FloatArray:
    """Axis by keyword."""
    return np.array([[1.0, 0.0] if "alimentos" in text else [0.0, 1.0] for text in texts])


def _corpus(tmp_path: Path) -> CorpusStorage:
  storage = CorpusStorage(tmp_path)
  storage.save_text(sample_text("a", "Cessa o dever de alimentos.", "Lei 1.111/2000, art. 1"))
  storage.save_text(sample_text("b", "Regras da tutela.", "Lei 1.111/2000, art. 2"))
  return storage


def test_meaning_finds_article_without_shared_words(tmp_path: Path) -> None:
  storage = _corpus(tmp_path)
  assert index_statutes(storage, FakeEncoder()) == len(storage.load_texts())
  hits = search_statutes(storage, "paro de pagar pensão?", 1, FakeEncoder())
  assert [hit.origin_ref for hit in hits] == ["Lei 1.111/2000, art. 1"]


def test_explicit_citation_comes_first(tmp_path: Path) -> None:
  storage = _corpus(tmp_path)
  index_statutes(storage, FakeEncoder())
  hits = search_statutes(storage, "pensão no art. 2 da Lei 1.111/2000", 2, FakeEncoder())
  assert hits[0].origin_ref == "Lei 1.111/2000, art. 2"


def test_falls_back_to_words(tmp_path: Path) -> None:
  storage = _corpus(tmp_path)
  assert search_statutes(storage, "tutela", 1)[0].doc_id == "b"
  assert search_statutes(storage, "tutela", 1, FakeEncoder())[0].doc_id == "b"
  assert load_encoder(tmp_path) is None


def test_reindex_replaces_vectors(tmp_path: Path) -> None:
  storage = _corpus(tmp_path)
  index_statutes(storage, FakeEncoder())
  index_statutes(storage, FakeEncoder())
  stored = storage.search_similar(np.array([1.0, 0.0]), MODEL_VERSION, 10)
  assert len(stored) == len(storage.load_texts())


def test_chunk_keeps_every_word() -> None:
  text = " ".join(str(number) for number in range(CHUNK_WORDS + 5))
  assert " ".join(chunk(text)) == text
  assert chunk("") == [""]
