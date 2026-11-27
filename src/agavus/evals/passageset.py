"""Frozen citation holdout. Each passage belongs to its paired primary reference.

Both domains include unavailable provisions and questions requiring facts outside
the corpus. Empty references mean the expected answer is a deferral.
"""

from agavus.evals.retrievalset import RetrievalItem
from agavus.taxonomy.areas import FAMILIA, MARIA_DA_PENHA


def _item(number: int, area: str, law: str, row: tuple[str, str, str]) -> RetrievalItem:
  article, text, passage = row
  refs = (f"{law}, art. {article}",) if article else ()
  return RetrievalItem(f"p{number:02}", text, refs, area, (passage,) if passage else ())


_FAMILY = (
  (
    "1.511",
    "No casamento, há igualdade entre os deveres do marido e os da esposa?",
    "igualdade de direitos e deveres dos cônjuges",
  ),
  (
    "1.512",
    "Quem declara pobreza paga a primeira certidão de casamento?",
    "primeira certidão serão isentos de selos, emolumentos e custas, para as pessoas cuja "
    "pobreza for declarada",
  ),
  (
    "1.514",
    "Em que momento o juiz declara formado o vínculo conjugal?",
    "manifestam, perante o juiz, a sua vontade de estabelecer vínculo conjugal, e o juiz "
    "os declara casados",
  ),
  (
    "1.517",
    "Para casar aos dezesseis anos basta a vontade do casal?",
    "dezesseis anos podem casar, exigindo-se autorização de ambos os pais, ou de seus "
    "representantes legais",
  ),
  (
    "1.519",
    "Se meus pais negaram sem motivo a autorização para casar, quem pode suprir a recusa?",
    "A denegação do consentimento, quando injusta, pode ser suprida pelo juiz",
  ),
  (
    "1.528",
    "Antes do casamento quem explica os diferentes regimes de bens aos noivos?",
    "dever do oficial do registro esclarecer os nubentes a respeito dos fatos que podem "
    "ocasionar a invalidade do casamento, bem como sobre os diversos regimes de bens",
  ),
  (
    "1.565",
    "Quando nos casarmos, eu posso acrescentar o sobrenome dela ao meu?",
    "Qualquer dos nubentes, querendo, poderá acrescer ao seu o sobrenome do outro",
  ),
  (
    "1.566",
    "O art. 1.566 do Código Civil menciona respeito entre os cônjuges?",
    "respeito e consideração mútuos",
  ),
  (
    "1.568",
    "Como se divide entre os cônjuges a despesa com sustento da família e educação dos filhos?",
    "na proporção de seus bens e dos rendimentos do trabalho, para o sustento da família e "
    "a educação dos filhos",
  ),
  (
    "1.569",
    "Quem escolhe onde o casal vai morar depois do casamento?",
    "O domicílio do casal será escolhido por ambos os cônjuges",
  ),
  (
    "1.579",
    "O divórcio muda os direitos dos pais em relação aos filhos?",
    "O divórcio não modificará os direitos e deveres dos pais em relação aos filhos",
  ),
  (
    "1.590",
    "As regras de guarda e de alimentos dos menores alcançam filhos maiores incapazes?",
    "guarda e prestação de alimentos aos filhos menores estendem-se aos maiores incapazes",
  ),
  (
    "1.591",
    "Como o Código Civil define parentes em linha reta?",
    "São parentes em linha reta as pessoas que estão umas para com as outras na relação de "
    "ascendentes e descendentes",
  ),
  (
    "1.592",
    "Parentes em linha colateral vão até qual grau?",
    "parentes em linha colateral ou transversal, até o quarto grau",
  ),
  (
    "1.596",
    "Um filho adotado tem os mesmos direitos que os demais filhos?",
    "ou por adoção, terão os mesmos direitos e qualificações",
  ),
  (
    "1.603",
    "No art. 1.603 do Código Civil, que documento prova a filiação?",
    "A filiação prova-se pela certidão do termo de nascimento registrada no Registro Civil",
  ),
  (
    "1.607",
    "Pai e mãe precisam reconhecer juntos o filho havido fora do casamento?",
    "O filho havido fora do casamento pode ser reconhecido pelos pais, conjunta ou separadamente",
  ),
  (
    "1.687",
    "Na separação de bens, quem administra o patrimônio de cada cônjuge?",
    "sob a administração exclusiva de cada um dos cônjuges",
  ),
  (
    "",
    "Qual foi o valor de alimentos que o juiz fixou ontem no meu processo de família?",
    "",
  ),
  (
    "",
    "Quero o texto do art. 9.876 do Código Civil sobre guarda dos filhos.",
    "",
  ),
)

_PROTECTION = (
  (
    "6",
    "A violência doméstica contra a mulher é classificada como violação de direitos humanos?",
    "violência doméstica e familiar contra a mulher constitui uma das formas de violação "
    "dos direitos humanos",
  ),
  (
    "10-A",
    "O atendimento pericial à vítima de violência doméstica deve funcionar sem interrupção?",
    "atendimento policial e pericial especializado, ininterrupto",
  ),
  (
    "11",
    "Em situação de violência doméstica quem acompanha a mulher para buscar seus pertences?",
    "acompanhar a ofendida para assegurar a retirada de seus pertences",
  ),
  (
    "14",
    "O juizado de violência doméstica pode praticar atos processuais de noite?",
    "Os atos processuais poderão realizar-se em horário noturno, conforme dispuserem as "
    "normas de organização judiciária",
  ),
  (
    "14-A",
    "A partilha de bens cabe ao juizado de violência doméstica no divórcio?",
    "Exclui-se da competência dos Juizados de Violência Doméstica e Familiar contra a "
    "Mulher a pretensão relacionada à partilha de bens",
  ),
  (
    "15",
    "Em processo cível da Lei Maria da Penha a ofendida pode escolher o juizado do lugar do fato?",
    "por opção da ofendida, para os processos cíveis regidos por esta Lei, o Juizado: I - "
    "do seu domicílio ou de sua residência; II - do lugar do fato",
  ),
  (
    "17",
    "Qual substituição de pena por multa o art. 17 da Lei Maria da Penha veda?",
    "substituição de pena que implique o pagamento isolado de multa",
  ),
  (
    "17-A",
    "O art. 17-A da Lei 11.340/2006 protege o nome da vítima?",
    "O nome da ofendida ficará sob sigilo",
  ),
  (
    "19",
    "É preciso esperar a audiência das partes antes de conceder medida protetiva de urgência?",
    "poderão ser concedidas de imediato, independentemente de audiência das partes",
  ),
  (
    "21",
    "Sou vítima de violência doméstica. Tenho de levar pessoalmente a intimação ao agressor?",
    "A ofendida não poderá entregar intimação ou notificação ao agressor",
  ),
  (
    "24",
    "A medida de proteção patrimonial pode suspender a procuração que a mulher deu ao agressor?",
    "suspensão das procurações conferidas pela ofendida ao agressor",
  ),
  (
    "25",
    "O Ministério Público participa da causa cível de violência doméstica mesmo sem ser parte?",
    "Ministério Público intervirá, quando não for parte, nas causas cíveis e criminais "
    "decorrentes da violência doméstica",
  ),
  (
    "26",
    "A fiscalização pelo Ministério Público abrange serviços particulares de atendimento à "
    "mulher em situação de violência doméstica?",
    "fiscalizar os estabelecimentos públicos e particulares de atendimento à mulher",
  ),
  (
    "28",
    "Na delegacia, a vítima de violência doméstica pode acessar assistência judiciária gratuita?",
    "Defensoria Pública ou de Assistência Judiciária Gratuita, nos termos da lei, em sede "
    "policial e judicial",
  ),
  (
    "29",
    "Quais áreas profissionais formam a equipe multidisciplinar do juizado de violência doméstica?",
    "profissionais especializados nas áreas psicossocial, jurídica e de saúde",
  ),
  (
    "30",
    "A equipe multidisciplinar pode apresentar laudos ao juiz em casos de violência doméstica?",
    "fornecer subsídios por escrito ao juiz, ao Ministério Público e à Defensoria Pública, "
    "mediante laudos",
  ),
  (
    "35",
    "A Lei Maria da Penha prevê casas-abrigos também para dependentes menores?",
    "casas-abrigos para mulheres e respectivos dependentes menores em situação de "
    "violência doméstica e familiar",
  ),
  (
    "38-A",
    "De quem é a responsabilidade de registrar a medida protetiva segundo o art. 38-A da "
    "Lei Maria da Penha?",
    "O juiz competente providenciará o registro da medida protetiva de urgência",
  ),
  (
    "",
    "Qual o número de vagas livres hoje na casa-abrigo para mulheres em situação de "
    "violência doméstica?",
    "",
  ),
  (
    "",
    "O que consta no art. 987 da Lei 11.340/2006 sobre medidas protetivas?",
    "",
  ),
)

FRESH = (
  *(_item(i, FAMILIA.code, "Lei 10.406/2002", row) for i, row in enumerate(_FAMILY, 1)),
  *(_item(i, MARIA_DA_PENHA.code, "Lei 11.340/2006", row) for i, row in enumerate(_PROTECTION, 21)),
)
