"""Strip personal data from legal text (LGPD; docs/corpus/privacy.md).

Conservative, pattern-based: CPF, CEP, email, phone, and names following
role markers. Process numbers and role tokens stay. Residual risk is
documented; processed text is never published.
"""

import re

_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_CEP = re.compile(r"\b\d{5}-\d{3}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?:\+?55[\s.-]?)?(?:\(?\d{2}\)?[\s.-]?)?(?:9\s?)?\d{4}[\s.-]?\d{4}\b")
_ROLE_NAME = re.compile(
  r"(?i)\b(autora?|réu?|requerente|requerid[oa]|vítima|ofendid[oa]|testemunha)"
  + r"(\s*[:\u2013-]\s*|\s+)([A-ZÀ-Ú][\wÀ-ú]+(?:\s+[A-ZÀ-Ú][\wÀ-ú]+)+)"
)
_LABELED_ID = re.compile(r"(?i)\b(RG|CPF|CNPJ)\s*[:nºo°.\s]*[\d./-]{5,20}")

_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
  (_LABELED_ID, r"\1: [removido]"),
  (_CPF, "[cpf removido]"),
  (_CEP, "[cep removido]"),
  (_EMAIL, "[email removido]"),
  (_PHONE, "[telefone removido]"),
  (_ROLE_NAME, r"\1\2[nome removido]"),
)


_URL = re.compile(r"https?://\S+")
_URL_TOKEN = "\x00URL{}\x00"


def anonymize(text: str) -> str:
  """Return text with personal data replaced by neutral tokens.

  URLs are protected first (digit sequences inside them are not phones).
  """
  urls: list[str] = []

  def protect(match: re.Match[str]) -> str:
    urls.append(match.group(0))
    return _URL_TOKEN.format(len(urls) - 1)

  protected = _URL.sub(protect, text)
  for pattern, replacement in _REPLACEMENTS:
    protected = pattern.sub(replacement, protected)
  for index, url in enumerate(urls):
    protected = protected.replace(_URL_TOKEN.format(index), url)
  return protected


def contains_personal_data(text: str) -> bool:
  """Scan for residual high-confidence personal data (CPF, email, phone)."""
  return any(pattern.search(text) for pattern in (_CPF, _EMAIL, _PHONE))
