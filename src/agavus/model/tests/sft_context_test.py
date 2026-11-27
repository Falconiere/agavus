"""Retrieval-in-context conversations over a synthetic index: format, negatives, provenance."""

from pathlib import Path

import pytest

from agavus.model.federal import PUBLIC_DOMAIN
from agavus.model.law_search import EMENTA, Hit, search
from agavus.model.sft import Conversation
from agavus.model.sft_context import (
  CONTEXT_CHARS,
  NOT_FOUND,
  SEARCHED,
  Deferrals,
  contextualize,
  contextualize_parallel,
  question_of,
  shown_hits,
  teacher_for_training,
  with_context,
)
from agavus.model.tests.law_search_test import LAW, NAME, OTHER, synthetic_index

GUARD = "Art. 2º A guarda dos filhos segue o acordo dos pais."
REPEATS = 200
NEGATIVES = 0.25
# How far the negative share may drift from NEGATIVES over REPEATS draws.
SLACK = 0.1


def _pair(
  question: str, answer: str, origin_ref: str, source: str = "rules:article"
) -> Conversation:
  return Conversation(
    (("user", question), ("assistant", answer)), source, origin_ref, PUBLIC_DOMAIN
  )


def _two_laws() -> list[Conversation]:
  """One article pair from each synthetic law."""
  return [
    _pair("Quem fica com a guarda dos filhos?", GUARD, f"{LAW}, art. 2"),
    _pair("Onde tramita o divórcio?", "Na vara.", f"{OTHER}, art. 1"),
  ]


@pytest.fixture
def db(tmp_path: Path) -> Path:
  return synthetic_index(tmp_path)


def test_with_context_numbers_the_articles_before_the_question() -> None:
  hits = [Hit("r1", NAME, "2º", GUARD), Hit("r2", "Lei X", "7", "y" * (CONTEXT_CHARS + 5))]
  assert with_context("E a guarda?", hits) == (
    f"[1] {NAME}, art. 2º:\n{GUARD}\n\n[2] Lei X, art. 7:\n{'y' * CONTEXT_CHARS}…\n\n"
    "Pergunta: E a guarda?"
  )
  assert with_context("Oi?", []) == "Pergunta: Oi?"
  ementa = Hit("r3", NAME, EMENTA, "Cria medidas.")
  assert with_context("Do que trata?", [ementa]).startswith(f"[1] {NAME}, ementa:\nCria")


def test_the_cited_article_is_in_context_even_when_search_misses_it(db: Path) -> None:
  hit = _pair("Quem fica com a guarda dos filhos?", GUARD, f"{LAW}, art. 2", "teacher:x")
  # Nothing in the question matches the cited article, so it is put among the hits.
  missed = _pair("Onde tramita o divórcio na vara de família?", "Na vara.", f"{LAW}, art. 3")
  found = contextualize([hit, missed], db, seed=0, negatives=0)
  assert [c.source for c in found] == ["teacher:x:context", "rules:article:context"]
  assert [(c.origin_ref, c.license) for c in found] == [
    (f"{LAW}, art. 2", PUBLIC_DOMAIN),
    (f"{LAW}, art. 3", PUBLIC_DOMAIN),
  ]
  question, answer = found[0].turns
  assert question == ("user", with_context(hit.turns[0][1], search(db, hit.turns[0][1])))
  assert answer == ("assistant", GUARD)
  context = found[1].turns[0][1]
  assert "[1] " in context and "[2] " in context and "[3] " not in context
  assert f"{NAME}, art. 3º:\nArt. 3º O juiz" in context
  assert search(db, missed.turns[0][1])[0].text in context


def test_negatives_defer_without_the_cited_article_and_repeat_for_a_seed(db: Path) -> None:
  pair = _pair("Guarda dos filhos, pensão, divórcio e bens?", GUARD, f"{LAW}, art. 2")
  found = contextualize([pair] * REPEATS, db, seed=7, negatives=NEGATIVES)
  assert found == contextualize([pair] * REPEATS, db, seed=7, negatives=NEGATIVES)
  negatives = [c for c in found if c.turns[1][1] == NOT_FOUND]
  assert abs(len(negatives) / REPEATS - NEGATIVES) < SLACK
  assert all(GUARD not in c.turns[0][1] and c.turns[0][1].count("[") == SEARCHED for c in negatives)
  assert all(GUARD in c.turns[0][1] for c in found if c.turns[1][1] != NOT_FOUND)
  assert all(c.origin_ref == pair.origin_ref for c in negatives)


def test_ementa_pairs_answer_from_the_ementa_and_unindexed_ones_are_skipped(db: Path) -> None:
  ementa = _pair("Do que trata a Lei nº 99.999?", "Cria medidas.", LAW, "rules:ementa")
  unindexed = _pair("Do que trata a outra lei?", "Divórcio.", OTHER, "rules:ementa")
  longer = Conversation((*ementa.turns, *ementa.turns), "s", f"{LAW}, art. 1", PUBLIC_DOMAIN)
  (found,) = contextualize([ementa, unindexed, longer], db, seed=0, negatives=0)
  assert found.turns[0][1].startswith(f"[1] {NAME}, ementa:\nCria medidas fictícias.")
  assert (found.turns[1][1], found.origin_ref) == ("Cria medidas.", LAW)


def test_unrelated_negatives_defer_on_another_laws_hits(db: Path) -> None:
  pairs = _two_laws() * (REPEATS // 2)
  found = contextualize(pairs, db, seed=3, negatives=0, unrelated=NEGATIVES)
  deferred = [(c, p) for c, p in zip(found, pairs, strict=True) if c.turns[1][1] == NOT_FOUND]
  assert abs(len(deferred) / REPEATS - NEGATIVES) < SLACK
  # LAW's articles are labeled with its NAME; OTHER, without a card, with its url.
  labels = {LAW: f"{NAME}, art.", OTHER: f"{OTHER}, art."}
  for conversation, pair in deferred:
    context = conversation.turns[0][1]
    assert labels[pair.origin_ref.partition(", ")[0]] not in context
    assert context.endswith(f"Pergunta: {pair.turns[0][1]}")


def test_parallel_shards_keep_every_pair_and_one_worker_is_serial(db: Path) -> None:
  pairs = _two_laws() * 10
  shares = Deferrals(negatives=0.3, unrelated=0.2)
  serial = contextualize(pairs, db, seed=5, negatives=0.3, unrelated=0.2)
  assert contextualize_parallel(pairs, db, seed=5, workers=1, shares=shares) == serial
  sharded = contextualize_parallel(pairs, db, seed=5, workers=2, shares=shares)
  assert sharded == contextualize_parallel(pairs, db, seed=5, workers=2, shares=shares)
  assert sorted(c.origin_ref for c in sharded) == sorted(c.origin_ref for c in serial)


def test_teacher_pairs_from_held_out_laws_are_dropped() -> None:
  held_out = [_pair("q", "a", f"{OTHER}, art. 1")]
  teacher = [_pair("q", "a", f"{LAW}, art. 2", "t"), _pair("q", "a", f"{OTHER}, art. 2", "t")]
  assert teacher_for_training(teacher, held_out) == teacher[:1]


def test_a_context_turn_reads_back_its_question_and_whole_articles(db: Path) -> None:
  question = "Do que trata a Lei nº 99.999 e a guarda?"
  hits = search(db, question)
  turn = with_context(question, hits)
  assert question_of(turn) == question and question_of("Oi?") == "Oi?"
  assert shown_hits(turn, db) == hits and any(hit.number == EMENTA for hit in hits)
  assert shown_hits(with_context("x", [Hit("r", "Lei Y", "9", "z")]), db) == []
