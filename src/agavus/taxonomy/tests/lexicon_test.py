"""Tests for the normalization lexicon (co-located)."""

from agavus.taxonomy.lexicon import normalize


def test_abbreviations_expanded() -> None:
  assert "lei maria da penha" in normalize("pedido com base na LMP")
  assert "medida protetiva" in normalize("requer MPU")


def test_synonyms_canonicalized() -> None:
  assert normalize("pensão alimentícia") == "alimentos"
  assert normalize("medida protetiva de urgência") == "medida protetiva"


def test_casefold_and_unknown_untouched() -> None:
  assert normalize("GUARDA") == "guarda"
  assert normalize("texto desconhecido") == "texto desconhecido"
