"""Retrieval question set v1: questions with the articles that answer them.

Authored and synthetic, safe in git; not reviewed by a lawyer. DEV may guide
ranking work. HIDDEN is opened once per milestone, never tuned against
(docs/evals.md). Questions never enter training or search data.
"""

from dataclasses import dataclass

RETRIEVAL_VERSION = "retrieval-v1"


@dataclass(frozen=True)
class RetrievalItem:
  """One question; any listed article is a correct answer."""

  id: str
  text: str
  relevant: tuple[str, ...]
  area: str = ""
  passages: tuple[str, ...] = ()


DEV: tuple[RetrievalItem, ...] = (
  RetrievalItem(
    "d01",
    "Qual artigo da lei Maria da Penha trata das medidas protetivas?",
    (
      "Lei 11.340/2006, art. 22",
      "Lei 11.340/2006, art. 23",
      "Lei 11.340/2006, art. 18",
      "Lei 11.340/2006, art. 19",
    ),
  ),
  RetrievalItem("d02", "Descumprir medida protetiva é crime?", ("Lei 11.340/2006, art. 24-A",)),
  RetrievalItem("d03", "O que é violência patrimonial?", ("Lei 11.340/2006, art. 7",)),
  RetrievalItem(
    "d04", "Que artigo define violência doméstica e familiar?", ("Lei 11.340/2006, art. 5",)
  ),
  RetrievalItem(
    "d05", "Em quanto tempo o juiz decide o pedido de proteção?", ("Lei 11.340/2006, art. 18",)
  ),
  RetrievalItem(
    "d06", "O agressor pode ser preso antes do julgamento?", ("Lei 11.340/2006, art. 20",)
  ),
  RetrievalItem(
    "d07", "Posso desistir da denúncia contra meu marido?", ("Lei 11.340/2006, art. 16",)
  ),
  RetrievalItem("d08", "O agressor pode pagar só cesta básica?", ("Lei 11.340/2006, art. 17",)),
  RetrievalItem("d09", "Meu nome fica em sigilo no processo?", ("Lei 11.340/2006, art. 17-A",)),
  RetrievalItem(
    "d10", "A mulher precisa de advogado em todos os atos?", ("Lei 11.340/2006, art. 27",)
  ),
  RetrievalItem(
    "d11", "O que a polícia deve fazer ao atender a vítima?", ("Lei 11.340/2006, art. 11",)
  ),
  RetrievalItem(
    "d12", "A polícia pode afastar o agressor do lar sem juiz?", ("Lei 11.340/2006, art. 12-C",)
  ),
  RetrievalItem(
    "d13",
    "Onde posso entrar com o processo, na minha cidade ou na dele?",
    ("Lei 11.340/2006, art. 15",),
  ),
  RetrievalItem(
    "d14",
    "O que o Ministério Público faz nesses casos?",
    (
      "Lei 11.340/2006, art. 26",
      "Lei 11.340/2006, art. 25",
    ),
  ),
  RetrievalItem("d15", "A mulher agredida tem prioridade no SUS?", ("Lei 11.340/2006, art. 9",)),
  RetrievalItem("d16", "O que é feminicídio?", ("Lei 14.994/2024, art. 2",)),
  RetrievalItem("d17", "O que é alienação parental?", ("Lei 12.318/2010, art. 2",)),
  RetrievalItem(
    "d18", "O que o juiz pode fazer contra quem aliena o filho?", ("Lei 12.318/2010, art. 6",)
  ),
  RetrievalItem(
    "d19", "O processo de alienação parental tem prioridade?", ("Lei 12.318/2010, art. 4",)
  ),
  RetrievalItem(
    "d20",
    "Como funciona a guarda compartilhada?",
    (
      "Lei 10.406/2002, art. 1.583",
      "Lei 10.406/2002, art. 1.584",
      "Lei 13.058/2014, art. 2",
    ),
  ),
  RetrievalItem("d21", "Quem tem direito de visitar os filhos?", ("Lei 10.406/2002, art. 1.589",)),
  RetrievalItem(
    "d22",
    "Casei de novo, perco meus filhos?",
    (
      "Lei 10.406/2002, art. 1.588",
      "Lei 10.406/2002, art. 1.636",
    ),
  ),
  RetrievalItem(
    "d23",
    "Quem deve alimentos entre parentes?",
    (
      "Lei 10.406/2002, art. 1.694",
      "Lei 10.406/2002, art. 1.696",
    ),
  ),
  RetrievalItem(
    "d24", "Como pedir revisão de pensão alimentícia?", ("Lei 10.406/2002, art. 1.699",)
  ),
  RetrievalItem(
    "d25", "Se minha ex casar de novo, paro de pagar pensão?", ("Lei 10.406/2002, art. 1.708",)
  ),
  RetrievalItem("d26", "Posso abrir mão da pensão?", ("Lei 10.406/2002, art. 1.707",)),
  RetrievalItem("d27", "Quem paga a pensão se o devedor morrer?", ("Lei 10.406/2002, art. 1.700",)),
  RetrievalItem("d28", "Quais os requisitos da união estável?", ("Lei 10.406/2002, art. 1.723",)),
  RetrievalItem("d29", "Qual o regime de bens da união estável?", ("Lei 10.406/2002, art. 1.725",)),
  RetrievalItem("d30", "O que é comunhão parcial de bens?", ("Lei 10.406/2002, art. 1.658",)),
  RetrievalItem(
    "d31",
    "O que acontece se o casal não escolher regime de bens?",
    ("Lei 10.406/2002, art. 1.640",),
  ),
  RetrievalItem(
    "d32",
    "Preciso da assinatura da minha esposa para vender a casa?",
    ("Lei 10.406/2002, art. 1.647",),
  ),
  RetrievalItem("d33", "Dá para divorciar sem dividir os bens?", ("Lei 10.406/2002, art. 1.581",)),
  RetrievalItem("d34", "Quais são os deveres dos cônjuges?", ("Lei 10.406/2002, art. 1.566",)),
  RetrievalItem(
    "d35",
    "Com que idade se pode casar?",
    (
      "Lei 10.406/2002, art. 1.517",
      "Lei 10.406/2002, art. 1.520",
    ),
  ),
  RetrievalItem("d36", "Filho adotivo tem os mesmos direitos?", ("Lei 10.406/2002, art. 1.596",)),
  RetrievalItem("d37", "Quando o pai perde o poder familiar?", ("Lei 10.406/2002, art. 1.638",)),
  RetrievalItem(
    "d38",
    "O que é bem de família?",
    (
      "Lei 10.406/2002, art. 1.712",
      "Lei 10.406/2002, art. 1.711",
    ),
  ),
  RetrievalItem(
    "d39",
    "Quem cuida da criança órfã?",
    (
      "Lei 10.406/2002, art. 1.728",
      "Lei 10.406/2002, art. 1.731",
    ),
  ),
  RetrievalItem("d40", "Quem pode ser colocado sob curatela?", ("Lei 10.406/2002, art. 1.767",)),
)

HIDDEN: tuple[RetrievalItem, ...] = (
  RetrievalItem("h01", "Que medidas o juiz pode impor ao agressor?", ("Lei 11.340/2006, art. 22",)),
  RetrievalItem(
    "h02",
    "Qual a pena para quem desobedece a ordem de não se aproximar da vítima?",
    ("Lei 11.340/2006, art. 24-A",),
  ),
  RetrievalItem(
    "h03", "Humilhar e controlar a mulher também é violência?", ("Lei 11.340/2006, art. 7",)
  ),
  RetrievalItem(
    "h04", "A lei vale para ex-namorado que nunca morou junto?", ("Lei 11.340/2006, art. 5",)
  ),
  RetrievalItem(
    "h05", "Quem pode pedir medidas protetivas de urgência?", ("Lei 11.340/2006, art. 19",)
  ),
  RetrievalItem("h06", "Vou ser avisada quando ele sair da prisão?", ("Lei 11.340/2006, art. 21",)),
  RetrievalItem(
    "h07", "Qual o prazo para representar contra o agressor?", ("Lei 11.340/2006, art. 16-A",)
  ),
  RetrievalItem(
    "h08",
    "A Lei dos Juizados Especiais se aplica à violência doméstica?",
    ("Lei 11.340/2006, art. 41",),
  ),
  RetrievalItem(
    "h09",
    "Posso pedir o divórcio no juizado de violência doméstica?",
    ("Lei 11.340/2006, art. 14-A",),
  ),
  RetrievalItem(
    "h10", "Quem não pode pagar tem direito à Defensoria Pública?", ("Lei 11.340/2006, art. 28",)
  ),
  RetrievalItem(
    "h11", "O que o delegado faz depois de registrar a ocorrência?", ("Lei 11.340/2006, art. 12",)
  ),
  RetrievalItem(
    "h12", "Tenho direito a ser atendida por policial mulher?", ("Lei 11.340/2006, art. 10-A",)
  ),
  RetrievalItem("h13", "Como o juiz protege os bens da mulher?", ("Lei 11.340/2006, art. 24",)),
  RetrievalItem(
    "h14",
    "O que faz a equipe multidisciplinar?",
    (
      "Lei 11.340/2006, art. 30",
      "Lei 11.340/2006, art. 29",
    ),
  ),
  RetrievalItem(
    "h15", "A medida protetiva fica registrada em banco de dados?", ("Lei 11.340/2006, art. 38-A",)
  ),
  RetrievalItem("h16", "Feminicídio é crime hediondo?", ("Lei 14.994/2024, art. 6",)),
  RetrievalItem(
    "h17", "Falar mal do pai para a criança tem nome na lei?", ("Lei 12.318/2010, art. 2",)
  ),
  RetrievalItem(
    "h18", "O juiz pode mandar fazer perícia psicológica na família?", ("Lei 12.318/2010, art. 5",)
  ),
  RetrievalItem(
    "h19",
    "Quem fica com a guarda quando um genitor atrapalha a convivência?",
    ("Lei 12.318/2010, art. 7",),
  ),
  RetrievalItem(
    "h20",
    "Pai e mãe não se entendem, o juiz dá qual guarda?",
    (
      "Lei 10.406/2002, art. 1.584",
      "Lei 13.058/2014, art. 2",
    ),
  ),
  RetrievalItem("h21", "Os avós podem visitar os netos?", ("Lei 10.406/2002, art. 1.589",)),
  RetrievalItem(
    "h22",
    "O divórcio muda os deveres dos pais com os filhos?",
    (
      "Lei 10.406/2002, art. 1.579",
      "Lei 10.406/2002, art. 1.632",
    ),
  ),
  RetrievalItem(
    "h23",
    "Posso cobrar pensão dos avós?",
    (
      "Lei 10.406/2002, art. 1.698",
      "Lei 10.406/2002, art. 1.696",
    ),
  ),
  RetrievalItem(
    "h24", "Perdi o emprego, posso diminuir a pensão?", ("Lei 10.406/2002, art. 1.699",)
  ),
  RetrievalItem(
    "h25", "O devedor casou de novo, a pensão acaba?", ("Lei 10.406/2002, art. 1.709",)
  ),
  RetrievalItem("h26", "A pensão é corrigida todo ano?", ("Lei 10.406/2002, art. 1.710",)),
  RetrievalItem(
    "h27", "O filho fora do casamento pode pedir pensão?", ("Lei 10.406/2002, art. 1.705",)
  ),
  RetrievalItem("h28", "Quais os deveres entre companheiros?", ("Lei 10.406/2002, art. 1.724",)),
  RetrievalItem(
    "h29", "Como converter união estável em casamento?", ("Lei 10.406/2002, art. 1.726",)
  ),
  RetrievalItem("h30", "A herança que recebi entra na partilha?", ("Lei 10.406/2002, art. 1.659",)),
  RetrievalItem(
    "h31", "Quem é obrigado a casar com separação de bens?", ("Lei 10.406/2002, art. 1.641",)
  ),
  RetrievalItem(
    "h32", "O pacto antenupcial precisa de escritura?", ("Lei 10.406/2002, art. 1.653",)
  ),
  RetrievalItem("h33", "Quando termina a sociedade conjugal?", ("Lei 10.406/2002, art. 1.571",)),
  RetrievalItem("h34", "Quem não pode casar?", ("Lei 10.406/2002, art. 1.521",)),
  RetrievalItem(
    "h35",
    "O casamento religioso vale no civil?",
    (
      "Lei 10.406/2002, art. 1.515",
      "Lei 10.406/2002, art. 1.516",
    ),
  ),
  RetrievalItem(
    "h36",
    "Posso voltar atrás no reconhecimento do filho?",
    (
      "Lei 10.406/2002, art. 1.609",
      "Lei 10.406/2002, art. 1.610",
    ),
  ),
  RetrievalItem("h37", "Quando o poder familiar acaba?", ("Lei 10.406/2002, art. 1.635",)),
  RetrievalItem(
    "h38", "O bem de família pode ser penhorado por dívidas?", ("Lei 10.406/2002, art. 1.715",)
  ),
  RetrievalItem("h39", "Quem não pode ser tutor?", ("Lei 10.406/2002, art. 1.735",)),
  RetrievalItem("h40", "O que é tomada de decisão apoiada?", ("Lei 10.406/2002, art. 1.783-A",)),
)
