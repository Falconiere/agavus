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


def test_letter_suffix_article() -> None:
  text = "Art. 24-A. Descumprir medida protetiva.\nArt. 25. Outro."
  articles = segment_articles(text)
  assert [number for number, _ in articles] == ["24-A", "25"]


def test_body_trimmed_at_chapter_heading() -> None:
  text = "Art. 24-A. Texto do artigo.\nCAPÍTULO III\nDA ATUAÇÃO\nArt. 25. Outro."
  articles = segment_articles(text)
  assert "CAPÍTULO" not in articles[0][1]
  assert "Texto do artigo" in articles[0][1]


def test_article_range_filter() -> None:
  html = (
    "<p>Art. 100. Fora.</p><p>Art. 1511. Família.</p><p>Art. 1783. Fim.</p><p>Art. 1800. Fora.</p>"
  )
  docs = parse_statute("Lei X", html, "raw/x.html", article_range=(1511, 1783))
  refs = [doc.origin_ref for doc in docs]
  assert refs == ["Lei X, art. 1511", "Lei X, art. 1783"]


def test_thousands_dot_article_numbers() -> None:
  html = "<p>Art. 1.511. Texto família.</p><p>Art. 1.783. Fim.</p>"
  docs = parse_statute("Lei 10.406/2002", html, "raw/x.html", article_range=(1511, 1783))
  assert [doc.origin_ref for doc in docs] == [
    "Lei 10.406/2002, art. 1.511",
    "Lei 10.406/2002, art. 1.783",
  ]


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


def test_decode_utf16_bom() -> None:
  payload = "Art. 24-A. Descumprir medida protetiva.".encode("utf-16")
  assert "Art. 24-A" in decode_html(payload)


def test_revoked_text_excluded() -> None:
  html = (
    "<p>Art. 24-A. Texto atual.</p>"
    '<p><span style="text-decoration:line-through">Pena revogada antiga.</span></p>'
    "<p><s>Riscado também.</s></p>"
  )
  text = html_to_text(html)
  assert "Texto atual" in text
  assert "revogada" not in text
  assert "Riscado" not in text


def test_ordinal_joined_and_cr_stripped() -> None:
  html = "<p>Art. 2</p><p>o</p><p>Texto\r\nda lei.</p>"
  text = html_to_text(html)
  assert "Art. 2o" in text
  assert "\r" not in text
