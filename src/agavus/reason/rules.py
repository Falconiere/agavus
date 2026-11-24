"""Versioned rule packs: requirements checklists as auditable data.

Rules fire only when ALL conditions have grounded facts; conclusions point
to verbatim-citable provisions. Human-reviewed, versioned (docs/reasoning.md).
"""

from dataclasses import dataclass

from agavus.taxonomy.areas import FAMILIA, MARIA_DA_PENHA

RULES_VERSION = "rules-v2"


@dataclass(frozen=True)
class Rule:
  """One syllogism major premise: conditions → cited conclusion."""

  id: str
  conditions: tuple[str, ...]
  conclusion_ref: str
  conclusion: str
  area: str = MARIA_DA_PENHA.code


RULES: tuple[Rule, ...] = (
  Rule(
    id="R-24A",
    conditions=("medida_vigente", "descumprimento"),
    conclusion_ref="Lei 11.340/2006, art. 24-A",
    conclusion=(
      "Descumprir medida protetiva de urgência é crime "
      "(art. 24-A, Lei 11.340/2006, redação da Lei 14.994/2024)."
    ),
    area=MARIA_DA_PENHA.code,
  ),
  Rule(
    id="R-22",
    conditions=("violencia_domestica", "contexto_familiar", "urgencia"),
    conclusion_ref="Lei 11.340/2006, art. 22",
    conclusion=(
      "Constatada violência doméstica e familiar contra a mulher em situação "
      "de urgência/risco, cabem medidas protetivas de urgência "
      "(art. 22, Lei 11.340/2006)."
    ),
    area=MARIA_DA_PENHA.code,
  ),
  Rule(
    id="R-AP",
    conditions=("campanha_desqualificacao", "obstaculo_convivencia"),
    conclusion_ref="Lei 12.318/2010, art. 2",
    conclusion=(
      "A desqualificação da conduta do genitor aliada ao obstáculo à convivência "
      "caracteriza indício de ato de alienação parental (art. 2º, Lei 12.318/2010)."
    ),
    area=FAMILIA.code,
  ),
)
