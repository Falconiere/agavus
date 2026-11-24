"""Tests for statute parsing (tests folder; public-domain-style fixture)."""

from agavus.corpus.records import TextSource, text_document_id
from agavus.corpus.statutes import (
  decode_html,
  extract_citations,
  html_to_text,
  parse_statute,
  segment_articles,
)

FIXTURE_HTML = """<html><body>
<p>LEI Nº 12.318, DE 26 DE AGOSTO DE 2010.</p>
<p>Art. 1º Esta Lei dispõe sobre a alienação parental.</p>
<p>Art. 2</p><p>o</p>
<p>Considera-se ato de alienação parental a interferência na formação
psicológica da criança ou do adolescente.</p>
<p>Art. 3º A alienação parental fere o direito fundamental da criança,
conforme o art. 227 da Constituição e a Lei 8.069/1990.</p>
</body></html>"""


def test_segment_articles_with_split_ordinal() -> None:
  articles = segment_articles(html_to_text(FIXTURE_HTML))
  numbers = [number for number, _ in articles]
  assert numbers == ["1", "2", "3"]
  assert "Considera-se ato de alienação parental" in articles[1][1]


def test_duplicate_article_first_wins() -> None:
  text = "Art. 8º Primeiro texto.\nArt. 8º Texto repetido.\nArt. 9º Fim."
  articles = segment_articles(text)
  assert [number for number, _ in articles] == ["8", "9"]
  assert "Primeiro texto" in articles[0][1]


def test_parse_statute_documents() -> None:
  expected_articles = 3
  docs = parse_statute("Lei 12.318/2010", FIXTURE_HTML, "raw/statutes/x.html")
  assert len(docs) == expected_articles
  assert docs[1].origin_ref == "Lei 12.318/2010, art. 2"
  assert docs[1].id == text_document_id(TextSource.STATUTE, docs[1].origin_ref)
  assert "Lei 8.069/1990" in docs[2].citations


def test_extract_citations() -> None:
  refs = extract_citations("conforme art. 22 da Lei 11.340/2006 e art. 5º")
  assert "art. 22" in refs
  assert "Lei 11.340/2006" in refs


def test_decode_cp1252_fallback() -> None:
  payload = "Considera-se alienação parental".encode("cp1252")
  assert "alienação" in decode_html(payload)
  assert decode_html(b"texto utf-8") == "texto utf-8"
