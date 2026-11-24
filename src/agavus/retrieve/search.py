"""BM25 retrieval + citation boost over the corpus (FTS5, zero-dep).

Lexical search via SQLite FTS5 bm25(); citation features boost documents
whose references match the query (design §6: citations are near-decisive
in legal text). Every hit carries its verbatim origin_ref and a verbatim
query-centered snippet — answers quote these, never generate.
"""

from dataclasses import dataclass

from agavus.corpus.statutes import extract_citations
from agavus.corpus.storage import CorpusStorage
from agavus.taxonomy.lexicon import WORD_RE, content_tokens, normalize

__all__ = ["WORD_RE", "Hit", "extract_snippet", "match_expression", "search"]

CITATION_BOOST = 2.0
ORIGIN_CITATION_BOOST = 3.0
SNIPPET_LEN = 200
OVERFETCH = 3


@dataclass(frozen=True)
class Hit:
  """One retrieved document, citable and auditable."""

  doc_id: str
  origin_ref: str
  score: float
  snippet: str
  citations: tuple[str, ...]


def match_expression(query: str) -> str:
  """Sanitized FTS5 query: deduplicated OR of quoted content tokens."""
  tokens = content_tokens(query) or WORD_RE.findall(normalize(query))
  unique = list(dict.fromkeys(tokens))
  return " OR ".join(f'"{token}"' for token in unique)


def extract_snippet(text: str, query: str, max_len: int = SNIPPET_LEN) -> str:
  """Verbatim substring of text up to max_len, shifted to cover query terms."""
  if len(text) <= max_len:
    return text
  lowered = text.casefold()
  for token in content_tokens(query):
    pos = lowered.find(token)
    if pos >= max_len // 2:
      start = min(max(0, pos - max_len // 4), len(text) - max_len)
      return text[start : start + max_len]
  return text[:max_len]


def search(storage: CorpusStorage, query: str, limit: int = 5) -> list[Hit]:
  """Ranked hits: BM25 + citation boost; empty query yields no hits."""
  expression = match_expression(query)
  if not expression:
    return []
  raw = storage.search_texts(expression, limit=limit * OVERFETCH)
  texts = storage.load_texts_by_ids([doc_id for doc_id, _ in raw])
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
        snippet=extract_snippet(document.text, query),
        citations=document.citations,
      )
    )
  hits.sort(key=lambda hit: -hit.score)
  return hits[:limit]
