"""BM25 retrieval + citation boost over the corpus (FTS5, zero-dep).

Lexical search via SQLite FTS5 bm25(); citation features boost documents
whose references match the query (design §6: citations are near-decisive
in legal text). Every hit carries its verbatim origin_ref — answers will
quote these, never generate (docs/response-quality.md).
"""

import re
from dataclasses import dataclass

from agavus.corpus.statutes import extract_citations
from agavus.corpus.storage import CorpusStorage
from agavus.taxonomy.lexicon import normalize

CITATION_BOOST = 2.0
ORIGIN_CITATION_BOOST = 3.0
SNIPPET_LEN = 200
OVERFETCH = 3
WORD_RE = re.compile(r"[a-zà-ú0-9]+")


@dataclass(frozen=True)
class Hit:
  """One retrieved document, citable and auditable."""

  doc_id: str
  origin_ref: str
  score: float
  snippet: str
  citations: tuple[str, ...]


def match_expression(query: str) -> str:
  """Sanitized FTS5 query: normalized OR of quoted tokens (no injection)."""
  tokens = WORD_RE.findall(normalize(query))
  return " OR ".join(f'"{token}"' for token in tokens)


def search(storage: CorpusStorage, query: str, limit: int = 5) -> list[Hit]:
  """Ranked hits: BM25 + citation boost; empty query yields no hits."""
  expression = match_expression(query)
  if not expression:
    return []
  raw = storage.search_texts(expression, limit=limit * OVERFETCH)
  texts = {document.id: document for document in storage.load_texts()}
  query_citations = set(extract_citations(query))
  hits: list[Hit] = []
  for doc_id, bm25_score in raw:
    document = texts.get(doc_id)
    if document is None:
      continue
    origin_cites = set(extract_citations(document.origin_ref))
    body_cites = set(document.citations)
    boost = ORIGIN_CITATION_BOOST * len(query_citations & origin_cites)
    boost += CITATION_BOOST * len(query_citations & body_cites)
    hits.append(
      Hit(
        doc_id=doc_id,
        origin_ref=document.origin_ref,
        score=-bm25_score + boost,
        snippet=document.text[:SNIPPET_LEN],
        citations=document.citations,
      )
    )
  hits.sort(key=lambda hit: -hit.score)
  return hits[:limit]
