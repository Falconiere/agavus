"""Authored supervision, never runtime negation or decision-verb rules.

Every wording template and all six scenario variants share a split group.
These fictional procedural facts cannot establish performance on real decisions.
"""

from agavus.classify.learned.types import Sample

CATEGORIES = (
  "direct_negation",
  "double_negation",
  "passive",
  "quotation",
  "conditional",
  "mixed_requests",
)
_TEMPLATES = (
  "O dispositivo registra que {decision}.",
  "Consta da decisão definitiva: {decision}.",
  "Ao encerrar o julgamento, o juízo declarou que {decision}.",
  "A conclusão expressa do pronunciamento é esta: {decision}.",
  "No resultado publicado, ficou estabelecido que {decision}.",
  "Depois de examinar os autos, a magistratura decidiu: {decision}.",
  "O capítulo conclusivo informa que {decision}.",
  "Foi proferida determinação final segundo a qual {decision}.",
  "A ata consigna como resultado do julgamento que {decision}.",
  "Na parte dispositiva, lê-se a conclusão: {decision}.",
  "O julgamento terminou com a seguinte resolução: {decision}.",
  "A decisão comunicada às partes estabelece que {decision}.",
  "Após a análise, o resultado judicial foi que {decision}.",
  "O pronunciamento resolveu a questão nestes termos: {decision}.",
  "O tribunal confirmou em seu dispositivo que {decision}.",
  "Do despacho final resulta expressamente que {decision}.",
  "A resolução judicial contém a conclusão de que {decision}.",
  "Em julgamento concluído, decidiu-se que {decision}.",
  "A comunicação do resultado esclarece que {decision}.",
  "O comando judicial já proferido determina que {decision}.",
)
_REQUESTS = (
  "afastamento do lar",
  "proibição de contato",
  "proteção patrimonial",
  "guarda provisória",
  "alimentos provisórios",
  "visitas supervisionadas",
  "divórcio",
  "partilha",
  "uso do imóvel",
  "alteração de guarda",
  "suspensão de visitas",
  "proteção de documentos",
  "separação de corpos",
  "revisão de alimentos",
  "regulamentação de visitas",
  "restituição de bens",
  "reconhecimento de união",
  "dissolução de união",
  "restrição de aproximação",
  "proteção de dependentes",
)
SCENARIO_VERSION = "synthetic-authored-v2"


def _passages(category: str, request: str) -> tuple[str, str, str]:
  subject = f"o pedido de {request}"
  yes = f"{subject} foi concedido"
  no = f"{subject} não foi concedido: foi indeferido"
  unknown = f"{subject} aguarda decisão e o mérito não foi examinado"
  if category == "double_negation":
    return (
      f"o juízo não deixou de conceder {subject}",
      f"não é verdade que o juízo não tenha negado {subject}; ele foi indeferido",
      f"não se pode afirmar nem negar o resultado; {unknown}",
    )
  if category == "passive":
    return f"foi concedido pelo juízo {subject}", f"foi negado pelo juízo {subject}", unknown
  if category == "quotation":
    quoted = f'a parte escreveu: "requeiro que {subject} seja concedido". Contudo, '
    return quoted + yes, quoted + no, quoted + unknown
  if category == "conditional":
    condition = f"se comprovada a necessidade, {subject} seria concedido. No julgamento efetivo, "
    return condition + yes, condition + no, f"se houver prova, {subject} será concedido"
  if category == "mixed_requests":
    other = "o pedido distinto de adiamento da audiência"
    return (
      f"{yes}, mas {other} foi negado",
      f"{no}, mas {other} foi concedido",
      f"{unknown}, mas {other} foi concedido",
    )
  return yes, no, unknown


def relation_samples() -> tuple[Sample, ...]:
  """120 scenario families, six pairs each, with exactly two examples per label."""
  samples: list[Sample] = []
  for index, (template, request) in enumerate(zip(_TEMPLATES, _REQUESTS, strict=True)):
    for category in CATEGORIES:
      group = f"relation:{category}:{index}"
      positive, negative, unknown = _passages(category, request)
      statements = (
        f"O pedido de {request} foi concedido.",
        f"O pedido de {request} foi negado.",
      )
      pairs = (
        (positive, 0, "support"),
        (negative, 1, "support"),
        (negative, 0, "contradiction"),
        (positive, 1, "contradiction"),
        (unknown, 0, "unknown"),
        (unknown, 1, "unknown"),
      )
      for offset, (passage, statement, label) in enumerate(pairs):
        samples.append(
          Sample(
            id=f"{group}:{offset}",
            text=template.format(decision=passage),
            label=label,
            group=group,
            origin_ref=f"synthetic:{group}",
            label_source=SCENARIO_VERSION,
            statement=statements[statement],
            category=category,
            template=f"template:{index}",
          )
        )
  return tuple(samples)
