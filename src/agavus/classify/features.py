"""Text → sparse TF-IDF features with legal mitigations (design §6.1).

Word 1-2 grams + char 3-grams (morphology), negation-scoped marks
(nega_protetiva ≠ protetiva), decision-verb features, citation flags,
lexicon normalization. Sparse rows: dict feature-index → weight.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from agavus.corpus.statutes import extract_citations
from agavus.taxonomy.lexicon import normalize

WORD_RE = re.compile(r"[a-zà-ú0-9]+")
NEG_WINDOW = 3
NEGATION_CUES = frozenset(
  {"não", "nunca", "jamais", "indefiro", "nego", "revogo", "improcedente", "desprovido"}
)
DECISION_VERBS = frozenset(
  {"defiro", "concedo", "homologo", "provido", "procedente", "indefiro", "nego", "revogo"}
)
STOPWORDS = frozenset(
  {"de", "da", "do", "das", "dos", "e", "o", "a", "os", "as", "um", "uma", "em", "no", "na"}
)


def tokenize(text: str) -> list[str]:
  """Normalized word tokens (lexicon applied first)."""
  return WORD_RE.findall(normalize(text))


def legal_tokens(text: str) -> list[str]:
  """All feature tokens for a text: words, bigrams, negation, verbs, citations."""
  words = tokenize(text)
  features = list(words)
  features += [f"{first}_{second}" for first, second in pairwise(words)]
  for index, word in enumerate(words):
    if word in NEGATION_CUES:
      for target in words[index + 1 : index + 1 + NEG_WINDOW]:
        if target not in STOPWORDS:
          features.append(f"nega_{target}")
    if word in DECISION_VERBS:
      features.append(f"verb_{word}")
  features += [f"cite_{ref.replace(' ', '_').replace('.', '')}" for ref in extract_citations(text)]
  return features


def _char_ngrams(text: str, size: int) -> list[str]:
  """Word-internal char n-grams (morphology), never across word boundaries."""
  grams: list[str] = []
  for word in tokenize(text):
    if len(word) >= size:
      grams += [f"c_{word[i : i + size]}" for i in range(len(word) - size + 1)]
  return grams


@dataclass(frozen=True)
class VectorizerConfig:
  """Vocabulary pruning and feature-family switches."""

  min_df: int = 2
  max_features: int = 20000
  char_sizes: tuple[int, ...] = (3,)


class TfidfVectorizer:
  """Fit-once vocabulary + IDF; transform to L2-normalized sparse rows."""

  def __init__(self, config: VectorizerConfig | None = None) -> None:
    self.config = config if config is not None else VectorizerConfig()
    self.vocab: dict[str, int] = {}
    self.idf: list[float] = []

  def _features(self, text: str) -> list[str]:
    features = legal_tokens(text)
    for size in self.config.char_sizes:
      features += _char_ngrams(text, size)
    return features

  def fit(self, texts: Sequence[str]) -> None:
    """Build vocabulary from document frequencies (train split only)."""
    df: Counter[str] = Counter()
    for text in texts:
      df.update(set(self._features(text)))
    keep = [term for term, count in df.items() if count >= self.config.min_df]
    keep.sort(key=lambda term: (-df[term], term))
    keep = keep[: self.config.max_features]
    self.vocab = {term: index for index, term in enumerate(keep)}
    total = len(texts)
    self.idf = [math.log((1 + total) / (1 + df[term])) + 1 for term in keep]

  def transform(self, texts: Sequence[str]) -> list[dict[int, float]]:
    """Sublinear TF x IDF, L2-normalized; unknown terms ignored."""
    rows: list[dict[int, float]] = []
    for text in texts:
      counts = Counter(self._features(text))
      row: dict[int, float] = {}
      for term, count in counts.items():
        index = self.vocab.get(term)
        if index is not None:
          row[index] = (1 + math.log(count)) * self.idf[index]
      norm = math.sqrt(sum(value * value for value in row.values())) or 1.0
      rows.append({index: value / norm for index, value in row.items()})
    return rows
