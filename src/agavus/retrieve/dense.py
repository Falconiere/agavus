"""Statute ranking by meaning, with explicit citations always first.

Falls back to BM25 when there is no encoder or no index, so the classical
core stays complete (docs/DESIGN.md §6, tier 4).
"""

from agavus.corpus.records import TextDocument, TextSource
from agavus.corpus.statutes import extract_citations
from agavus.corpus.storage import CorpusStorage
from agavus.retrieve.encoder import MODEL_VERSION, Encoder
from agavus.retrieve.search import Hit, extract_snippet, search

CHUNK_WORDS = 120
CHUNK_SEPARATOR = "#"
# Articles have several chunks, so fetch more vectors than articles wanted.
CHUNK_OVERFETCH = 20


def chunk(text: str) -> list[str]:
  """Consecutive pieces of at most CHUNK_WORDS words."""
  words = text.split()
  starts = range(0, max(1, len(words)), CHUNK_WORDS)
  return [" ".join(words[start : start + CHUNK_WORDS]) for start in starts]


def index_statutes(storage: CorpusStorage, encoder: Encoder) -> int:
  """Replace the vectors of all statutes; returns the number of articles."""
  documents = storage.load_texts_by_source(TextSource.STATUTE)
  owners: list[str] = []
  passages: list[str] = []
  for document in documents:
    for number, piece in enumerate(chunk(document.text)):
      owners.append(f"{document.id}{CHUNK_SEPARATOR}{number}")
      passages.append(piece)
  if passages:
    vectors = encoder.encode_passages(passages)
    storage.replace_embeddings(MODEL_VERSION, list(zip(owners, vectors, strict=True)))
  return len(documents)


def _hit(document: TextDocument, query: str, score: float) -> Hit:
  return Hit(
    doc_id=document.id,
    origin_ref=document.origin_ref,
    score=score,
    snippet=extract_snippet(document.text, query),
    citations=document.citations,
  )


def _cited(storage: CorpusStorage, query: str) -> list[Hit]:
  """Articles the question names by law and number."""
  refs = extract_citations(query)
  laws = [ref for ref in refs if ref.startswith("Lei ")]
  articles = [ref for ref in refs if ref.startswith("art. ")]
  found = (
    storage.load_text_by_origin_ref(f"{law}, {article}") for law in laws for article in articles
  )
  return [_hit(document, query, 1.0) for document in found if document is not None]


def search_statutes(
  storage: CorpusStorage, query: str, limit: int, encoder: Encoder | None = None
) -> list[Hit]:
  """Best statute articles: cited ones, then by meaning, else by words."""
  cited = _cited(storage, query)
  if encoder is None or not query.strip():
    ranked = search(storage, query, limit, TextSource.STATUTE)
  else:
    ranked = _by_meaning(storage, query, limit, encoder)
  seen = {hit.doc_id for hit in cited}
  return [*cited, *(hit for hit in ranked if hit.doc_id not in seen)][:limit]


def _by_meaning(storage: CorpusStorage, query: str, limit: int, encoder: Encoder) -> list[Hit]:
  vector = encoder.encode_queries([query])[0]
  best: dict[str, float] = {}
  for key, score in storage.search_similar(vector, MODEL_VERSION, limit * CHUNK_OVERFETCH):
    best.setdefault(key.split(CHUNK_SEPARATOR)[0], score)
  if not best:
    return search(storage, query, limit, TextSource.STATUTE)
  texts = storage.load_texts_by_ids(list(best))
  return [_hit(texts[doc_id], query, score) for doc_id, score in best.items() if doc_id in texts][
    :limit
  ]
