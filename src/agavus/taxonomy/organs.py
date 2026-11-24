"""Weak-label rules from órgão julgador names (versioned, auditable).

Policy (Jev-validated 2026-11-24, docs/external-validation.md): Tribunal do
Júri without TPU subjects goes to REVIEW, never guessed — feminicídio would
be M, common homicide O.
"""

import re

from agavus.taxonomy.areas import FAMILIA, MARIA_DA_PENHA, OUT_OF_SCOPE

ORGAN_RULES_VERSION = "organs-v1"
_DOMESTIC = re.compile(r"violência doméstica", re.IGNORECASE)
_FAMILY = re.compile(r"\bfamília\b", re.IGNORECASE)
_CRIMINAL = re.compile(r"\bvaras? crimina(?:l|is)\b", re.IGNORECASE)
_JURI = re.compile(r"tribunal do júri", re.IGNORECASE)


def organ_label(orgao: str) -> str | None:
  """Weak area signal from the organ name; None = no signal or needs review."""
  if _DOMESTIC.search(orgao):
    return MARIA_DA_PENHA.code
  if _JURI.search(orgao):
    return None
  if _FAMILY.search(orgao):
    return FAMILIA.code
  if _CRIMINAL.search(orgao):
    return OUT_OF_SCOPE.code
  return None
