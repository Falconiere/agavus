"""Tests for verified chains (tests folder, synthetic corpus)."""

from pathlib import Path

from agavus.corpus.httpfixtures import sample_text
from agavus.corpus.storage import CorpusStorage
from agavus.reason.chain import build_chain

QUERY = "ele descumpriu a medida protetiva, o que acontece?"


def _corpus(tmp_path: Path, with_provision: bool) -> CorpusStorage:
  storage = CorpusStorage(tmp_path)
  if with_provision:
    storage.save_text(
      sample_text(
        doc_id="p24a",
        origin_ref="Lei 11.340/2006, art. 24-A",
        text="Descumprir decisão judicial que defere medidas protetivas: pena reclusão.",
      )
    )
  return storage


def test_complete_chain_with_premise(tmp_path: Path) -> None:
  chain = build_chain(QUERY, _corpus(tmp_path, with_provision=True))
  assert chain is not None
  assert chain.complete
  assert chain.rule_id == "R-24A"
  assert chain.premises[0].ref == "Lei 11.340/2006, art. 24-A"
  assert all(fact.span in QUERY for fact in chain.facts)


def test_broken_chain_without_premise(tmp_path: Path) -> None:
  chain = build_chain(QUERY, _corpus(tmp_path, with_provision=False))
  assert chain is not None
  assert not chain.complete
  assert chain.premises == ()


def test_no_rule_no_chain(tmp_path: Path) -> None:
  assert build_chain("quero pedir divórcio", _corpus(tmp_path, True)) is None
