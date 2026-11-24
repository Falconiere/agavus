"""Tests for anonymization (tests folder, fully synthetic personal data)."""

from agavus.corpus.anonymize import anonymize, contains_personal_data


def test_cpf_removed() -> None:
  labeled = anonymize("CPF 123.456.789-09 da parte")
  assert "123.456.789-09" not in labeled
  assert "[removido]" in labeled
  bare = anonymize("documento 123.456.789-09 anexado")
  assert "[cpf removido]" in bare
  assert not contains_personal_data(labeled)
  assert not contains_personal_data(bare)


def test_email_and_phone_removed() -> None:
  text = "contato: fulano@example.com, (61) 99999-8888"
  cleaned = anonymize(text)
  assert "[email removido]" in cleaned
  assert "[telefone removido]" in cleaned
  assert not contains_personal_data(cleaned)


def test_role_name_removed_role_kept() -> None:
  cleaned = anonymize("Autor: João Silva dos Santos requer a guarda.")
  assert "João Silva" not in cleaned
  assert "Autor" in cleaned
  assert "[nome removido]" in cleaned


def test_process_number_kept() -> None:
  text = "Processo 0701234-56.2024.8.07.0001"
  assert "0701234-56.2024.8.07.0001" in anonymize(text)


def test_clean_text_untouched() -> None:
  text = "Considera-se ato de alienação parental a interferência na formação psicológica."
  assert anonymize(text) == text
  assert not contains_personal_data(text)


def test_urls_protected() -> None:
  text = "Ver http://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/Lei/L13641.htm#art2"
  assert anonymize(text) == text
