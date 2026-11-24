"""Tests for organ weak-label rules (co-located, synthetic organ names)."""

from agavus.taxonomy.organs import organ_label


def test_domestic_violence_vara_is_m() -> None:
  assert organ_label("1ª Vara de Violência Doméstica e Familiar contra a Mulher") == "M"


def test_family_vara_is_f() -> None:
  assert organ_label("2ª Vara de Família e Sucessões") == "F"


def test_criminal_vara_is_o() -> None:
  assert organ_label("3ª Vara Criminal") == "O"


def test_juri_without_subjects_needs_review() -> None:
  assert organ_label("Tribunal do Júri") is None


def test_unknown_organ_no_signal() -> None:
  assert organ_label("Vara Cível") is None
