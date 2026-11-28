"""Articles the chat may quote: the domain statutes by meaning, then all Planalto law by words.

The Planalto index leaves the exam statutes out (Lei 11.340, 12.318, 13.058, 14.994 and the
Civil Code) so the language-model exam stays clean. The chat takes those from the product's
corpus instead, ranked by the product's E5 statute search, and cites them by their Planalto
url and name like every other article. Training never reads this module: retrieval only
shows text to the reader.
"""

from pathlib import Path

from agavus.corpus.statutes import STATUTES
from agavus.corpus.storage import CorpusStorage
from agavus.model.law_search import Hit, search, unique_hits
from agavus.model.planalto import PLANALTO_DIR, SITE
from agavus.model.sft_rules import article_number, law_card
from agavus.retrieve.dense import search_statutes
from agavus.retrieve.encoder import Encoder

# Articles taken from each side: the domain statutes and the whole Planalto index.
DOMAIN_HITS = 3
PLANALTO_HITS = 3


def statute_names(data_dir: Path) -> dict[str, tuple[str, str]]:
  """Each statute's corpus name ('Lei 11.340/2006') → (Planalto url, citable name).

  The citable name comes from the statute's Planalto capture; without one, the corpus name.
  """
  folder = data_dir / PLANALTO_DIR
  names = {}
  for ref in STATUTES:
    capture = folder / "sources" / f"{ref.url.removeprefix(SITE)}.html"
    card = law_card(ref.url, capture.read_bytes()) if capture.is_file() else None
    names[ref.name] = (ref.url, ref.name if card is None else card.name)
  return names


class ChatSearch:
  """Domain statutes from the corpus (when present) and Planalto articles from the index."""

  def __init__(
    self, data_dir: Path, db: Path, storage: CorpusStorage | None, encoder: Encoder | None
  ) -> None:
    self._db, self._storage, self._encoder = db, storage, encoder
    self._names = statute_names(data_dir)

  def _domain(self, question: str) -> list[Hit]:
    if self._storage is None:
      return []
    found = []
    for hit in search_statutes(self._storage, question, DOMAIN_HITS, self._encoder):
      document = self._storage.load_text_by_origin_ref(hit.origin_ref)
      law, _, number = hit.origin_ref.partition(", art. ")
      if document is None or law not in self._names:
        continue
      url, name = self._names[law]
      text = document.text
      found.append(Hit(f"{url}, art. {number}", name, article_number(hit.origin_ref, text), text))
    return found

  def hits(self, question: str) -> list[Hit]:
    """Domain articles first, then Planalto ones, each cited once."""
    return unique_hits([*self._domain(question), *search(self._db, question, PLANALTO_HITS)])
