"""Extractive fact extraction: every fact grounded by a verbatim query span.

Conservative patterns + negation guard (design §6.1 #1). Facts are boolean
presence flags; a fact without a span never exists.
"""

import re
from dataclasses import dataclass

FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
  (
    "medida_vigente",
    re.compile(
      r"medida protetiva(?: de urgência)?(?: vigente| em vigor)?|"
      r"descumpriu a medida protetiva|descumprimento de medida protetiva",
      re.IGNORECASE,
    ),
  ),
  ("descumprimento", re.compile(r"descumpriu|descumprimento|violou a medida", re.IGNORECASE)),
  (
    "violencia_domestica",
    re.compile(
      r"violência doméstica|me agrediu|agressão|me ameaçou|ameaça de morte|"
      r"violência psicológica|violência patrimonial",
      re.IGNORECASE,
    ),
  ),
  (
    "contexto_familiar",
    re.compile(
      r"marido|esposo|companheiro|ex-marido|ex-namorado|namorado|pai dos meus filhos",
      re.IGNORECASE,
    ),
  ),
  ("urgencia", re.compile(r"urgente|urgência|imediat|risco|tenho medo|com medo", re.IGNORECASE)),
  (
    "campanha_desqualificacao",
    re.compile(
      r"fala mal|desqualifica|fazendo a cabeça|campanha de desqualificação",
      re.IGNORECASE,
    ),
  ),
  (
    "obstaculo_convivencia",
    re.compile(
      r"impede (?:a )?(?:visita|convivência)|dificulta (?:a )?(?:visita|convivência)|"
      r"não deixa (?:eu |o pai |a mãe )?ver (?:o |a |meu |minha )?(?:filh[oa]|criança)",
      re.IGNORECASE,
    ),
  ),
)
NEG_PREFIX = re.compile(r"(?:não|nunca|jamais)\s+$", re.IGNORECASE)


@dataclass(frozen=True)
class Fact:
  """One grounded fact: field, presence, and the verbatim span proving it."""

  field: str
  value: bool
  span: str


def extract_facts(query: str) -> tuple[Fact, ...]:
  """Facts present in the query, each with its grounding span."""
  facts: list[Fact] = []
  for field, pattern in FACT_PATTERNS:
    match = pattern.search(query)
    if match and not NEG_PREFIX.search(query[: match.start()]):
      facts.append(Fact(field=field, value=True, span=match.group(0)))
  return tuple(facts)
