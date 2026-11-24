"""Tests for the normalization lexicon (co-located)."""

from agavus.taxonomy.lexicon import content_tokens, normalize


def test_abbreviations_expanded() -> None:
  assert "lei maria da penha" in normalize("pedido com base na LMP")
  assert "medida protetiva" in normalize("requer MPU")
  assert "violência patrimonial" in normalize("sofreu VP do companheiro")


def test_word_boundaries_preserve_substrings() -> None:
  assert normalize("computador e impugnação") == "computador e impugnação"


def test_synonyms_canonicalized() -> None:
  assert normalize("pensão alimentícia") == "alimentos"
  assert normalize("medida protetiva de urgência") == "medida protetiva"
  assert normalize("medidas protetivas de urgência") == "medida protetiva"
  assert normalize("Lei 11.340/2006") == "lei maria da penha"


def test_casefold_and_unknown_untouched() -> None:
  assert normalize("GUARDA") == "guarda"
  assert normalize("texto desconhecido") == "texto desconhecido"


def test_content_tokens_strip_stopwords() -> None:
  assert content_tokens("qual é o artigo da lei de guarda") == ["artigo", "lei", "guarda"]
