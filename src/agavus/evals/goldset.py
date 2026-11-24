"""Author-made routing gold set v1 — synthetic, safe in git.

Seeded from the Jev external-validation pilot (docs/external-validation.md).
Firewall: gold data never enters training (docs/evals.md).
"""

from dataclasses import dataclass

GOLD_VERSION = "routing-v1"


@dataclass(frozen=True)
class GoldItem:
  """One authored query with its expected routing label."""

  id: str
  text: str
  expected: str


ROUTING_V1: tuple[GoldItem, ...] = (
  GoldItem("r01", "Quero pedir o divórcio e a partilha dos bens do casal.", "F"),
  GoldItem("r02", "Como faço para fixar a pensão alimentícia do meu filho?", "F"),
  GoldItem(
    "r03", "Ele me ameaçou de morte e quebrou meu celular, preciso de medida protetiva.", "M"
  ),
  GoldItem(
    "r04", "Ele descumpriu a medida protetiva de afastamento do lar, o que acontece agora?", "M"
  ),
  GoldItem(
    "r05",
    "Ele não devolve as crianças no horário combinado e acho que está fazendo alienação parental.",
    "F",
  ),
  GoldItem(
    "r06", "Meu ex-marido me persegue e me ameaça, e também não paga a pensão das crianças.", "M"
  ),
  GoldItem("r07", "Fui preso em flagrante por tráfico de drogas, qual é a pena?", "O"),
  GoldItem("r08", "Fui demitido sem justa causa, quais verbas rescisórias tenho direito?", "O"),
  GoldItem("r09", "meu ex não deixa eu ver meu filho", "F"),
  GoldItem(
    "r10", "Sofro violência psicológica do meu esposo, ele controla meu dinheiro e me humilha.", "M"
  ),
)
