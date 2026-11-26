"""Authored query scenarios; paired variants share a calibration fold.

Synthetic examples only. No evaluation questions or case-derived text.
"""

from agavus.taxonomy.areas import FAMILIA, MARIA_DA_PENHA, OUT_OF_SCOPE

SCENARIOS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
  (
    "support-ascendant",
    FAMILIA.code,
    (
      "Minha avó não consegue se sustentar; os netos podem pagar alimentos?",
      "Meus irmãos precisam dividir comigo o sustento do nosso pai idoso?",
    ),
  ),
  (
    "name-divorce",
    FAMILIA.code,
    (
      "Depois do divórcio, como atualizar meus documentos com meu nome anterior?",
      "Posso manter o nome de casada depois da separação judicial?",
    ),
  ),
  (
    "child-relocation",
    FAMILIA.code,
    (
      "Os pais discordam sobre em qual cidade o menino vai morar após o divórcio.",
      "Quais autorizações os responsáveis precisam para mudar o domicílio da criança?",
    ),
  ),
  (
    "contact-schedule",
    FAMILIA.code,
    (
      "A visita foi cancelada várias semanas seguidas, quero regularizar a convivência.",
      "O acordo de convivência não está sendo seguido pelo responsável pela criança.",
    ),
  ),
  (
    "support-adult",
    FAMILIA.code,
    (
      "Meu filho já trabalha, posso pedir revisão dos alimentos que pago?",
      "A filha entrou na universidade; como fica a obrigação alimentar dos pais?",
    ),
  ),
  (
    "parentage-register",
    FAMILIA.code,
    (
      "Como incluir os dois pais socioafetivos no registro civil da criança?",
      "Descobri uma divergência sobre a filiação no registro do meu filho.",
    ),
  ),
  (
    "marital-agreement",
    FAMILIA.code,
    (
      "Precisamos homologar nosso acordo sobre os bens do casamento.",
      "É possível ajustar a partilha de um apartamento depois do divórcio?",
    ),
  ),
  (
    "grandparent-contact",
    FAMILIA.code,
    (
      "Os avós querem regulamentar judicialmente as visitas ao neto.",
      "A avó pode pedir convivência com a neta após a separação dos pais?",
    ),
  ),
  (
    "dv-child-handover",
    MARIA_DA_PENHA.code,
    (
      "O ex-companheiro usa as visitas da filha para me agredir.",
      "Na troca da guarda, meu ex me empurrou e disse que me mataria.",
    ),
  ),
  (
    "dv-order-contact",
    MARIA_DA_PENHA.code,
    (
      "A proibição de contato com meu ex está em vigor e ele me mandou mensagens.",
      "O pai do meu bebê violou a medida protetiva e me esperou no portão.",
    ),
  ),
  (
    "dv-refuge",
    MARIA_DA_PENHA.code,
    (
      "Preciso sair com segurança da casa onde meu companheiro me agride.",
      "Meu ex me ameaçou e preciso de acolhimento longe dele esta noite.",
    ),
  ),
  (
    "dv-financial-control",
    MARIA_DA_PENHA.code,
    (
      "Meu marido reteve meu cartão e não me deixa comprar comida.",
      "Minha companheira fica com meu pagamento e me ameaça se eu reclamar.",
    ),
  ),
  (
    "dv-tracking",
    MARIA_DA_PENHA.code,
    (
      "Meu ex-namorado segue meus trajetos mesmo depois de eu pedir para parar.",
      "Meu companheiro instalou um rastreador para controlar onde estou.",
    ),
  ),
  (
    "dv-documents",
    MARIA_DA_PENHA.code,
    (
      "Meu esposo esconde meus documentos para impedir que eu deixe a relação.",
      "Meu parceiro destruiu minhas coisas quando anunciei a separação.",
    ),
  ),
  (
    "dv-family-threats",
    MARIA_DA_PENHA.code,
    (
      "Minha irmã é ameaçada de morte pelo companheiro dentro de casa.",
      "Minha filha adulta está sendo agredida pelo namorado.",
    ),
  ),
  (
    "dv-coercion",
    MARIA_DA_PENHA.code,
    (
      "Meu companheiro me obriga a ter relações mesmo quando recuso.",
      "Meu ex usa fotos íntimas para me obrigar a reatar o relacionamento.",
    ),
  ),
  (
    "dv-removal",
    MARIA_DA_PENHA.code,
    (
      "A polícia afastou meu agressor de casa, mas meu marido voltou e arrombou a porta.",
      "Meu ex desobedeceu o afastamento determinado na medida protetiva.",
    ),
  ),
  (
    "dv-custody-intimidation",
    MARIA_DA_PENHA.code,
    (
      "Meu marido ameaça me bater se eu pedir a guarda das crianças.",
      "Meu ex me intimida usando os filhos para impedir que eu denuncie a violência.",
    ),
  ),
  (
    "work-supervision",
    OUT_OF_SCOPE.code,
    (
      "O gerente me constrangeu na frente da equipe por causa das metas.",
      "Meu supervisor berrou comigo durante o expediente; posso reclamar na Justiça do Trabalho?",
    ),
  ),
  (
    "work-threat",
    OUT_OF_SCOPE.code,
    (
      "Um colega sem relação familiar comigo ameaçou me bater na fábrica.",
      "O patrão me humilha no trabalho e ameaça descontar meu salário.",
    ),
  ),
  (
    "public-robbery",
    OUT_OF_SCOPE.code,
    (
      "Dois desconhecidos armados levaram minha carteira na saída do metrô.",
      "Um ladrão tomou meu relógio na parada de ônibus.",
    ),
  ),
  (
    "shop-conflict",
    OUT_OF_SCOPE.code,
    (
      "Um consumidor sem vínculo familiar comigo ameaçou o caixa do supermercado.",
      "Um comprador disse que iria me agredir depois de discutir sobre um produto.",
    ),
  ),
  (
    "phone-repair",
    OUT_OF_SCOPE.code,
    (
      "A assistência técnica devolveu meu telefone sem consertar o defeito.",
      "O celular novo parou de funcionar durante o prazo de garantia.",
    ),
  ),
  (
    "health-contract",
    OUT_OF_SCOPE.code,
    (
      "A operadora do convênio se recusa a autorizar meu exame.",
      "Paguei o plano de saúde, mas o hospital recusou a internação coberta pelo contrato.",
    ),
  ),
  (
    "lease-debt",
    OUT_OF_SCOPE.code,
    (
      "Meu locatário está há quatro meses sem depositar o aluguel.",
      "Aluguei um imóvel comercial e a empresa deixou parcelas em aberto.",
    ),
  ),
  (
    "noise-building",
    OUT_OF_SCOPE.code,
    (
      "O salão vizinho faz festas até de madrugada e não consigo descansar.",
      "O condomínio pode multar o morador por barulho fora do horário?",
    ),
  ),
  (
    "criminal-appeal",
    OUT_OF_SCOPE.code,
    (
      "Meu recurso criminal é sobre uma condenação por receptação.",
      "Estou respondendo por furto em uma loja e quero entender o procedimento penal.",
    ),
  ),
  (
    "traffic-conflict",
    OUT_OF_SCOPE.code,
    (
      "Um motorista desconhecido me ameaçou depois de uma colisão.",
      "Fui agredido por outro passageiro durante uma discussão no transporte público.",
    ),
  ),
  (
    "bank-service",
    OUT_OF_SCOPE.code,
    (
      "O banco bloqueou meu dinheiro por erro no aplicativo.",
      "Uma cobrança de empréstimo apareceu sem que eu tivesse contratado nada.",
    ),
  ),
  (
    "school-service",
    OUT_OF_SCOPE.code,
    (
      "A escola particular cobra uma multa que não estava no contrato.",
      "A faculdade perdeu meus documentos e não libera a matrícula.",
    ),
  ),
  (
    "property-dispute",
    OUT_OF_SCOPE.code,
    (
      "Meu vizinho invadiu parte do terreno com uma construção.",
      "A imobiliária não devolveu o depósito de garantia do apartamento.",
    ),
  ),
  (
    "stranger-stalking",
    OUT_OF_SCOPE.code,
    (
      "Um desconhecido sem vínculo familiar me persegue na saída do trabalho.",
      "Uma pessoa estranha que conheci na internet passou a me ameaçar.",
    ),
  ),
)
