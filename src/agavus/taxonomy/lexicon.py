"""Synonym/abbreviation normalization lexicon (design §6.1 #2).

Curated and versioned; applied at preprocessing and query expansion.
Word-boundary regex prevents short abbreviations (LMP, MPU, VP) from
corrupting longer words (e.g. computador, impugnação).
"""

import re

LEXICON_VERSION = "lexicon-v2"
WORD_RE = re.compile(r"[a-zà-ú0-9]+")
STOPWORDS: frozenset[str] = frozenset(
  {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "contra",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "ela",
    "ele",
    "em",
    "entre",
    "é",
    "essa",
    "esse",
    "esta",
    "está",
    "este",
    "eu",
    "foi",
    "já",
    "mais",
    "mas",
    "me",
    "meu",
    "minha",
    "na",
    "nas",
    "no",
    "nos",
    "num",
    "numa",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "qual",
    "quais",
    "quando",
    "que",
    "quem",
    "se",
    "sem",
    "ser",
    "seu",
    "seus",
    "sobre",
    "sua",
    "suas",
    "são",
    "também",
    "tem",
    "um",
    "uma",
    "uns",
    "umas",
  }
)
# Words that frame a question without naming its subject.
QUERY_FRAMING: frozenset[str] = frozenset(
  {"art", "artigo", "artigos", "diz", "existe", "lei", "leis", "número", "tem", "trata"}
)
SYNONYMS: dict[str, str] = {
  "lmp": "lei maria da penha",
  "lei da maria da penha": "lei maria da penha",
  "lei 11.340/2006": "lei maria da penha",
  "lei 11.340/06": "lei maria da penha",
  "lei 11.340": "lei maria da penha",
  "lei maria da penha": "lei maria da penha",
  "pensão alimentícia": "alimentos",
  "pensao alimenticia": "alimentos",
  "medida protetiva de urgência": "medida protetiva",
  "medidas protetivas de urgência": "medida protetiva",
  "medidas protetivas": "medida protetiva",
  "mpu": "medida protetiva",
  "vp": "violência patrimonial",
  "alienação parental": "alienacao parental",
}
_SYNONYM_RE = re.compile(
  r"\b(?:" + "|".join(re.escape(term) for term in sorted(SYNONYMS, key=len, reverse=True)) + r")\b"
)


def normalize(text: str) -> str:
  """Casefold and map whole-word synonyms/abbreviations to canonical terms."""
  return _SYNONYM_RE.sub(lambda match: SYNONYMS[match.group(0)], text.casefold())


def content_tokens(text: str) -> list[str]:
  """Normalized word tokens with Portuguese stopwords removed."""
  return [token for token in WORD_RE.findall(normalize(text)) if token not in STOPWORDS]
