"""Statute ingestion: Planalto consolidated texts, article-level units.

Statutes are not windowed (docs/corpus/data-range.md): current consolidated
text, version-stamped by fetch date. Public-domain content.
"""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from agavus.corpus.records import LabelSource, TextDocument, TextSource, text_document_id
from agavus.taxonomy.lexicon import normalize

SOURCE = "statutes"
# Case-sensitive: articles open with "Art."; a line starting "art. 129 do ..." is a reference.
ART_MARKER = re.compile(r"(?m)^\s*Art\.?\s*(\d+(?:\.\d{3})*(?:-[A-Za-z])?)\s*[ºo°]?")
ART_REF = re.compile(r"\bart\.?\s*(\d+(?:\.\d{3})*(?:-[A-Za-z])?)\s*[ºo°]?", re.IGNORECASE)
HEADING = re.compile(r"(?m)^\s*(?:CAPÍTULO|TÍTULO|SEÇÃO|SUBSEÇÃO)\b", re.IGNORECASE)
LAW_REF = re.compile(r"\bLei\s+(?:n[ºo°.]?\s*)?(\d{1,3}(?:\.\d{3})*/\d{2,4})", re.IGNORECASE)
# Planalto pads some UTF-16 pages with single-byte spaces: each pair decodes to a dagger.
MISDECODED_PADDING = re.compile(r"\u2020{2,}\ufffd?\s*\Z")
LAW_ALIASES = {
  "lei maria da penha": "Lei 11.340/2006",
  "alienação parental": "Lei 12.318/2010",
  "código civil": "Lei 10.406/2002",
}
LAW_POPULAR_NAMES = {
  "Lei 11.340/2006": "lei maria da penha violência doméstica",
  "Lei 12.318/2010": "lei de alienação parental",
  "Lei 10.406/2002": "código civil família",
  "Lei 13.058/2014": "guarda compartilhada",
  "Lei 14.994/2024": "feminicídio",
}


# Names that say which law, not which subject: asking for "the article" needs a topic.
TOPICLESS_NAMES = ("lei maria da penha", "código civil")


def popular_name(origin_ref: str) -> str:
  """Searchable popular name for a law (index-only; citations stay verbatim)."""
  for prefix, name in LAW_POPULAR_NAMES.items():
    if origin_ref.startswith(prefix):
      return name
  return ""


@dataclass(frozen=True)
class StatuteRef:
  """One statute source: citable name + consolidated-text URL.

  article_range optionally keeps only articles in [start, end] (e.g. the
  Civil Code's família book) — scope discipline for large codes.
  """

  name: str
  url: str
  article_range: tuple[int, int] | None = None


STATUTES: tuple[StatuteRef, ...] = (
  StatuteRef(
    name="Lei 11.340/2006",
    url="https://www.planalto.gov.br/ccivil_03/_ato2004-2006/2006/lei/l11340.htm",
  ),
  StatuteRef(
    name="Lei 12.318/2010",
    url="https://www.planalto.gov.br/ccivil_03/_ato2007-2010/2010/lei/l12318.htm",
  ),
  StatuteRef(
    name="Lei 13.058/2014",
    url="https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2014/lei/l13058.htm",
  ),
  StatuteRef(
    name="Lei 14.994/2024",
    url="https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2024/lei/l14994.htm",
  ),
  StatuteRef(
    name="Lei 10.406/2002",
    url="https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
    article_range=(1511, 1783),
  ),
)


def _decode(payload: bytes) -> str:
  if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
    return payload.decode("utf-16", errors="replace")
  try:
    return payload.decode("utf-8-sig")
  except UnicodeDecodeError:
    return payload.decode("cp1252", errors="replace")


def decode_html(payload: bytes) -> str:
  """Planalto serves cp1252, UTF-8, and UTF-16 (Word pages); detect BOMs.

  Trailing padding is dropped, also from transcriptions that already carry it.
  """
  return MISDECODED_PADDING.sub("", _decode(payload))


def html_to_text(html: str) -> str:
  """Strip markup; drop revoked (line-through) text; keep line structure."""
  soup = BeautifulSoup(html, "html.parser")
  for tag in soup.find_all(["s", "strike", "del"]):
    tag.decompose()
  for tag in soup.find_all(style=re.compile(r"line-through", re.IGNORECASE)):
    tag.decompose()
  text = soup.get_text("\n", strip=True).replace("\r", "")
  return re.sub(r"(?m)^(Art\.?\s*[\d.\-A-Za-z]+)\n([ºo°])$", r"\1\2", text)


def _own_markers(text: str) -> list[re.Match[str]]:
  """Article markers of this law; ones inside “…” belong to a law it amends."""
  own: list[re.Match[str]] = []
  start = 0
  for marker in ART_MARKER.finditer(text):
    since = text[start : marker.start()]
    if since.count("“") > since.count("”"):
      continue
    own.append(marker)
    start = marker.start()
  return own


def segment_articles(text: str) -> list[tuple[str, str]]:
  """Split into (article number, body); first occurrence per number wins."""
  markers = _own_markers(text)
  seen: set[str] = set()
  articles: list[tuple[str, str]] = []
  for index, marker in enumerate(markers):
    number = marker.group(1)
    if number in seen:
      continue
    seen.add(number)
    end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
    body = text[marker.start() : end]
    heading = HEADING.search(body)
    if heading:
      body = body[: heading.start()]
    body = re.sub(r"\n(?!(?:§|Art\.|CAPÍTULO|TÍTULO|SEÇÃO|SUBSEÇÃO))", " ", body)
    articles.append((number, body.strip()))
  return articles


def extract_citations(text: str) -> tuple[str, ...]:
  """Normalized references found in text (articles, laws, named-law aliases)."""
  refs = {f"art. {number}" for number in ART_REF.findall(text)}
  refs |= {f"Lei {number}" for number in LAW_REF.findall(text)}
  lowered = normalize(text)
  refs |= {canonical for alias, canonical in LAW_ALIASES.items() if normalize(alias) in lowered}
  return tuple(sorted(refs))


def parse_statute(
  name: str, html: str, raw_ref: str, article_range: tuple[int, int] | None = None
) -> list[TextDocument]:
  """One TextDocument per article, with a verbatim-citable origin_ref."""
  text = html_to_text(html)
  documents: list[TextDocument] = []
  for number, body in segment_articles(text):
    if article_range is not None:
      base = int(number.replace(".", "").split("-")[0])
      if not article_range[0] <= base <= article_range[1]:
        continue
    ref = f"{name}, art. {number}"
    documents.append(
      TextDocument(
        id=text_document_id(TextSource.STATUTE, ref),
        source=TextSource.STATUTE,
        origin_ref=ref,
        published_at=None,
        text=body,
        citations=extract_citations(body),
        domain_label=None,
        label_source=LabelSource.PENDING,
        raw_ref=raw_ref,
      )
    )
  return documents
