"""Tests for retrieval (tests folder, synthetic corpus, no network)."""

from pathlib import Path

from agavus.corpus.records import LabelSource, TextDocument, TextSource
from agavus.corpus.storage import CorpusStorage
from agavus.retrieve.search import match_expression, search


def _corpus(tmp_path: Path) -> CorpusStorage:
  storage = CorpusStorage(tmp_path)
  docs = [
    TextDocument(
      id="d1",
      source=TextSource.STATUTE,
      origin_ref="Lei 11.340/2006, art. 22",
      published_at=None,
      text="A medida protetiva de urgência poderá ser concedida pelo juiz.",
      citations=("art. 22",),
      domain_label=None,
      label_source=LabelSource.PENDING,
      raw_ref="raw/x.html",
    ),
    TextDocument(
      id="d2",
      source=TextSource.STATUTE,
      origin_ref="Lei 12.318/2010, art. 2",
      published_at=None,
      text="Considera-se ato de alienação parental a interferência psicológica.",
      citations=(),
      domain_label=None,
      label_source=LabelSource.PENDING,
      raw_ref="raw/y.html",
    ),
  ]
  for document in docs:
    storage.save_text(document)
  storage.rebuild_index()
  return storage


def test_bm25_ranks_relevant_first(tmp_path: Path) -> None:
  hits = search(_corpus(tmp_path), "alienação parental")
  assert hits and hits[0].origin_ref == "Lei 12.318/2010, art. 2"


def test_citation_boost(tmp_path: Path) -> None:
  hits = search(_corpus(tmp_path), "medida protetiva art. 22 da Lei 11.340/2006")
  assert hits[0].doc_id == "d1"
  assert hits[0].score > 0


def test_query_sanitization(tmp_path: Path) -> None:
  assert match_expression('protetiva" OR (guarda') == '"protetiva" OR "or" OR "guarda"'
  hits = search(_corpus(tmp_path), 'protetiva"; DROP TABLE texts;--')
  assert hits and hits[0].doc_id == "d1"


def test_empty_query_no_hits(tmp_path: Path) -> None:
  assert search(_corpus(tmp_path), "!!! ???") == []


def test_hits_are_citable(tmp_path: Path) -> None:
  hit = search(_corpus(tmp_path), "medida protetiva")[0]
  assert hit.origin_ref
  assert hit.snippet
