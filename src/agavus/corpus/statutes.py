"""Statute ingestion: Planalto consolidated texts, article-level units.

Statutes are not windowed (docs/corpus/data-range.md): current consolidated
text, version-stamped by fetch date. Public-domain content.
"""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from agavus.corpus.records import LabelSource, TextDocument, TextSource, text_document_id

SOURCE = "statutes"
ART_MARKER = re.compile(r"(?m)^\s*Art\.?\s*(\d+)\s*[ºo°]?", re.IGNORECASE)
ART_REF = re.compile(r"\bart\.?\s*(\d+)\s*[ºo°]?", re.IGNORECASE)
LAW_REF = re.compile(r"\bLei\s+(?:n[ºo°.]?\s*)?(\d{1,3}(?:\.\d{3})*/\d{2,4})", re.IGNORECASE)


@dataclass(frozen=True)
class StatuteRef:
  """One statute source: citable name + consolidated-text URL."""

  name: str
  url: str


STATUTES: tuple[StatuteRef, ...] = (
  StatuteRef(
    name="Lei 11.340/2006",
    url="https://www.planalto.gov.br/ccivil_03/_ato2004-2006/2006/lei/l11340.htm",
  ),
  StatuteRef(
    name="Lei 12.318/2010",
    url="https://www.planalto.gov.br/ccivil_03/_ato2007-2010/2010/lei/l12318.htm",
  ),
)


def decode_html(payload: bytes) -> str:
  """Planalto serves windows-1252; prefer UTF-8 when valid."""
  try:
    return payload.decode("utf-8")
  except UnicodeDecodeError:
    return payload.decode("cp1252", errors="replace")


def html_to_text(html: str) -> str:
  """Strip markup, keeping line structure for article segmentation."""
  return BeautifulSoup(html, "html.parser").get_text("\n", strip=True)


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
    articles.append((number, text[marker.start() : end].strip()))
  return articles


def extract_citations(text: str) -> tuple[str, ...]:
  """Normalized references found in text (articles and laws)."""
  refs = {f"art. {number}" for number in ART_REF.findall(text)}
  refs |= {f"Lei {number}" for number in LAW_REF.findall(text)}
  return tuple(sorted(refs))


def parse_statute(name: str, html: str, raw_ref: str) -> list[TextDocument]:
  """One TextDocument per article, with a verbatim-citable origin_ref."""
  text = html_to_text(html)
  documents: list[TextDocument] = []
  for number, body in segment_articles(text):
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
