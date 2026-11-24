"""Statute ingestion: Planalto consolidated texts, article-level units.

Statutes are not windowed (docs/corpus/data-range.md): current consolidated
text, version-stamped by fetch date. Public-domain content.
"""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from agavus.corpus.records import LabelSource, TextDocument, TextSource, text_document_id

SOURCE = "statutes"
ART_MARKER = re.compile(r"(?m)^\s*Art\.?\s*(\d+(?:\.\d{3})*(?:-[A-Za-z])?)\s*[ºo°]?", re.IGNORECASE)
ART_REF = re.compile(r"\bart\.?\s*(\d+(?:\.\d{3})*(?:-[A-Za-z])?)\s*[ºo°]?", re.IGNORECASE)
HEADING = re.compile(r"(?m)^\s*(?:CAPÍTULO|TÍTULO|SEÇÃO|SUBSEÇÃO)\b", re.IGNORECASE)
LAW_REF = re.compile(r"\bLei\s+(?:n[ºo°.]?\s*)?(\d{1,3}(?:\.\d{3})*/\d{2,4})", re.IGNORECASE)


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


def decode_html(payload: bytes) -> str:
  """Planalto serves cp1252, UTF-8, and UTF-16 (Word pages); detect BOMs."""
  if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
    return payload.decode("utf-16", errors="replace")
  try:
    return payload.decode("utf-8-sig")
  except UnicodeDecodeError:
    return payload.decode("cp1252", errors="replace")


def html_to_text(html: str) -> str:
  """Strip markup; drop revoked (line-through) text; keep line structure."""
  soup = BeautifulSoup(html, "html.parser")
  for tag in soup.find_all(["s", "strike", "del"]):
    tag.decompose()
  for tag in soup.find_all(style=re.compile(r"line-through", re.IGNORECASE)):
    tag.decompose()
  text = soup.get_text("\n", strip=True).replace("\r", "")
  return re.sub(r"(?m)^(Art\.?\s*[\d.\-A-Za-z]+)\n([ºo°])$", r"\1\2", text)


def segment_articles(text: str) -> list[tuple[str, str]]:
  """Split into (article number, body); first occurrence per number wins."""
  markers = list(ART_MARKER.finditer(text))
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
  """Normalized references found in text (articles and laws)."""
  refs = {f"art. {number}" for number in ART_REF.findall(text)}
  refs |= {f"Lei {number}" for number in LAW_REF.findall(text)}
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
