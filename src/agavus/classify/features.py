"""Text → sparse TF-IDF features with legal mitigations (design §6.1).

Word 1-2 grams + char 3-grams (morphology), negation-scoped marks
(nega_protetiva ≠ protetiva), decision-verb features, citation flags,
citation x verb crosses, lexicon normalization. Sparse rows: dict index → weight.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from agavus.corpus.statutes import extract_citations
from agavus.taxonomy.lexicon import STOPWORDS, WORD_RE, normalize

NEG_WINDOW = 3
NEGATION_CUES = frozenset(
  {
    "não",
    "nao",
    "nunca",
    "jamais",
    "indefiro",
    "indeferiu",
    "indeferido",
    "indeferida",
    "indeferir",
    "nego",
    "negou",
    "negado",
    "negar",
    "revogo",
    "revogou",
    "revogado",
    "revogada",
    "improcedente",
    "desprovido",
    "desprovida",
  }
)
DECISION_VERBS = frozenset(
  {
    "defiro",
    "deferiu",
    "deferido",
    "deferida",
    "deferir",
    "concedo",
    "concedeu",
    "concedido",
    "concedida",
    "conceder",
    "homologo",
    "homologou",
    "homologado",
    "homologada",
    "provido",
    "provida",
    "procedente",
    "indefiro",
    "indeferiu",
    "indeferido",
    "indeferida",
    "indeferir",
    "nego",
    "negou",
    "negado",
    "negar",
    "revogo",
    "revogou",
    "revogado",
    "revogada",
  }
)


def tokenize(text: str) -> list[str]:
  """Normalized word tokens (lexicon applied first)."""
  return WORD_RE.findall(normalize(text))


def legal_tokens(text: str, words: Sequence[str] | None = None) -> list[str]:
  """All feature tokens: words, bigrams, negation, verbs, citations, crosses."""
  tokens = list(words) if words is not None else tokenize(text)
  features = list(tokens)
  features += [f"{first}_{second}" for first, second in pairwise(tokens)]
  verbs: list[str] = []
  for index, word in enumerate(tokens):
    if word in NEGATION_CUES:
      for target in tokens[index + 1 : index + 1 + NEG_WINDOW]:
        if target not in STOPWORDS:
          features.append(f"nega_{target}")
    if word in DECISION_VERBS:
      verb = f"verb_{word}"
      features.append(verb)
      verbs.append(verb)
  cites = [f"cite_{ref.replace(' ', '_').replace('.', '')}" for ref in extract_citations(text)]
  features += cites
  if cites and verbs:
    features += [f"cross_{cite}_{verb}" for cite in cites for verb in verbs]
  return features


def _char_ngrams(words: Sequence[str], size: int) -> list[str]:
  """Word-internal char n-grams (morphology), never across word boundaries."""
  grams: list[str] = []
  for word in words:
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
    words = tokenize(text)
    features = legal_tokens(text, words)
    for size in self.config.char_sizes:
      features += _char_ngrams(words, size)
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
