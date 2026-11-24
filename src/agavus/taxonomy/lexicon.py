"""Synonym/abbreviation normalization lexicon (design §6.1 #2).

Curated and versioned; applied at preprocessing and query expansion.
Seed set — grows with reviewed corpus evidence.
"""

LEXICON_VERSION = "lexicon-v1"
SYNONYMS: dict[str, str] = {
  "lmp": "lei maria da penha",
  "lei da maria da penha": "lei maria da penha",
  "lei 11.340": "lei maria da penha",
  "lei maria da penha": "lei maria da penha",
  "pensão alimentícia": "alimentos",
  "medida protetiva de urgência": "medida protetiva",
  "mpu": "medida protetiva",
  "alienação parental": "alienacao parental",
}


def normalize(text: str) -> str:
  """Casefold and map known synonyms/abbreviations to canonical terms."""
  lowered = text.casefold()
  for term, canonical in SYNONYMS.items():
    lowered = lowered.replace(term, canonical)
  return lowered
