"""Tests for fact extraction (tests folder, synthetic queries)."""

from agavus.reason.facts import extract_facts


def test_descumprimento_facts_grounded() -> None:
  facts = extract_facts("ele descumpriu a medida protetiva de afastamento")
  fields = {fact.field for fact in facts}
  assert {"medida_vigente", "descumprimento"} <= fields
  for fact in facts:
    assert fact.span
    assert fact.value


def test_span_is_verbatim() -> None:
  query = "meu marido me agrediu, tenho medo"
  facts = extract_facts(query)
  assert all(fact.span in query for fact in facts)


def test_negation_guard() -> None:
  facts = extract_facts("ele não descumpriu a medida protetiva")
  assert "descumprimento" not in {fact.field for fact in facts}


def test_no_facts_in_neutral_query() -> None:
  assert extract_facts("qual é o horário de funcionamento do fórum") == ()


def test_alienacao_parental_facts_grounded() -> None:
  query = "a mãe não deixa o pai ver a criança e fala mal dele"
  facts = extract_facts(query)
  assert {"campanha_desqualificacao", "obstaculo_convivencia"} <= {fact.field for fact in facts}
  assert all(fact.span in query for fact in facts)
