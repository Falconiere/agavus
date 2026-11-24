"""Tests for feature extraction (tests folder, synthetic texts)."""

import math

from agavus.classify.features import TfidfVectorizer, VectorizerConfig, legal_tokens, tokenize


def test_negation_scoping() -> None:
  tokens = legal_tokens("indefiro a protetiva")
  assert "nega_protetiva" in tokens
  assert "verb_indefiro" in tokens
  assert "nega_protetiva" in legal_tokens("o juiz indeferiu a medida protetiva")
  assert "nega_protetiva" not in legal_tokens("defiro a protetiva")


def test_citation_flags_and_crosses() -> None:
  tokens = legal_tokens("defiro a medida com base no art. 22 da Lei 11.340/2006")
  assert any(token.startswith("cite_") for token in tokens)
  assert any(token.startswith("cross_cite_") for token in tokens)


def test_lexicon_applied() -> None:
  assert "alimentos" in tokenize("pensão alimentícia")


def test_bigrams_present() -> None:
  assert "medida_protetiva" in legal_tokens("medida protetiva")


def test_char_ngrams_word_internal() -> None:
  vectorizer = TfidfVectorizer(VectorizerConfig(min_df=1, char_sizes=(3,)))
  vectorizer.fit(["protetiva urgente", "protetiva deferida"])
  assert "c_pro" in vectorizer.vocab
  assert not any(" " in term for term in vectorizer.vocab)


def test_vectorizer_fit_transform() -> None:
  texts = ["medida protetiva urgente", "medida protetiva deferida", "guarda compartilhada"]
  vectorizer = TfidfVectorizer(VectorizerConfig(min_df=2, char_sizes=()))
  vectorizer.fit(texts)
  rows = vectorizer.transform(texts)
  assert "medida" in vectorizer.vocab
  assert "guarda" not in vectorizer.vocab  # df=1 pruned
  assert rows[2] == {}  # fully-pruned document yields a zero vector
  for row in rows[:2]:
    norm = math.sqrt(sum(value * value for value in row.values()))
    tolerance = 1e-9
    assert abs(norm - 1.0) < tolerance


def test_unknown_terms_ignored() -> None:
  vectorizer = TfidfVectorizer(VectorizerConfig(min_df=1, char_sizes=()))
  vectorizer.fit(["alpha beta"])
  rows = vectorizer.transform(["alpha desconhecido"])
  assert len(rows[0]) == 1
