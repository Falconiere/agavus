"""Offline chat evaluation over a synthetic index and run: verdicts, kinds, the CLI report."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from agavus.model.agathe_run import save_run
from agavus.model.bundle import BUNDLE_FILE, Texts, digest, write_bundle
from agavus.model.chat_eval import (
  CORRECT,
  DEFERRED,
  LEAKED,
  MISSED,
  WRONG,
  ChatCase,
  judge,
  outcome,
  run_eval_chat,
  summary,
)
from agavus.model.federal import PUBLIC_DOMAIN
from agavus.model.report import SAMPLE_CHARS, SAMPLE_SEED
from agavus.model.sft import Conversation, save_conversations
from agavus.model.sft_context import NOT_FOUND, contextualize
from agavus.model.tests.generation_test import TOKENIZER, silent
from agavus.model.tests.law_search_test import LAW, NAME, synthetic_index
from agavus.model.tokenizer import TOKENIZER_FILE, save_tokenizer

GUARD = "Art. 2º A guarda dos filhos segue o acordo dos pais."
QUOTED = f"{GUARD}\n\nFonte: {NAME}, art. 2º."


def _pairs() -> list[Conversation]:
  """An answerable article pair and an ementa pair from the synthetic law."""
  ask = (("user", "Quem fica com a guarda dos filhos?"), ("assistant", QUOTED))
  ementa = (("user", "Do que trata a Lei nº 99.999?"), ("assistant", "Cria medidas fictícias."))
  return [
    Conversation(ask, "rules:article", f"{LAW}, art. 2", PUBLIC_DOMAIN),
    Conversation(ementa, "rules:ementa", LAW, PUBLIC_DOMAIN),
  ]


def test_verdicts_compare_the_shown_answer_with_the_gold_one() -> None:
  assert outcome(QUOTED, QUOTED) == CORRECT
  assert outcome(f"{GUARD[:40]}\n\nFonte: {NAME}, art. 2º.", QUOTED) == CORRECT
  assert outcome(f"{GUARD}\n\nFonte: {NAME}, art. 3º.", QUOTED) == WRONG
  assert outcome("Outro texto qualquer aqui.", QUOTED) == WRONG
  assert outcome(NOT_FOUND, QUOTED) == MISSED
  assert outcome(NOT_FOUND, NOT_FOUND) == DEFERRED
  assert outcome(QUOTED, NOT_FOUND) == LEAKED


def test_cases_know_their_kind_and_whether_the_check_blocked_them() -> None:
  case = ChatCase("rules:article:context", "r", "q", QUOTED, "inventado", NOT_FOUND, False, MISSED)
  assert case.kind == "article" and case.blocked
  negative = ChatCase(
    "teacher:x:context", "r", "q", NOT_FOUND, NOT_FOUND, NOT_FOUND, False, DEFERRED
  )
  assert negative.kind == "teacher/negative" and not negative.blocked
  lines = summary([case, negative])
  assert lines[-2] == (
    "answerable 1: correct 0, deferred 1, wrong shown 0; negatives 1: answered 0"
  )
  assert lines[-1] == "grounding check blocked 1 written answers (0 loops)"


def test_a_silent_run_defers_everything(tmp_path: Path) -> None:
  db = synthetic_index(tmp_path)
  found = contextualize(_pairs(), db, seed=0, negatives=0)
  cases = [judge(silent(4096), conversation, db) for conversation in found]
  assert [case.outcome for case in cases] == [MISSED, MISSED]
  assert [case.question for case in cases] == [pair.turns[0][1] for pair in _pairs()]
  assert not any(case.blocked for case in cases)


def _saved_run(data_dir: Path) -> Path:
  """A tiny chat run whose bundle and tokenizer sit where the CLI defaults look."""
  model = silent(4096)
  sha = write_bundle(data_dir / BUNDLE_FILE, Texts(("only held out", "um papel", "papel")))
  save_tokenizer(TOKENIZER, data_dir / TOKENIZER_FILE)
  record: dict[str, object] = {
    "config": asdict(model.config),
    "data_sha256": sha,
    "tokenizer_sha256": digest(data_dir / TOKENIZER_FILE),
    "sample": model.sample(seed=SAMPLE_SEED, max_chars=SAMPLE_CHARS),
    "chat": True,
  }
  save_run(data_dir / "run", model, record)
  return data_dir / "run"


def test_cli_judges_the_held_out_file_and_writes_every_case(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  monkeypatch.setenv("AGAVUS_DATA_DIR", str(tmp_path))
  db = synthetic_index(tmp_path)
  held_out = tmp_path / "context-eval.jsonl"
  save_conversations(held_out, contextualize(_pairs(), db, seed=0, negatives=0))
  run = _saved_run(tmp_path)
  assert run_eval_chat(["--run", str(run), "--eval", str(held_out), "--search", str(db)]) == 0
  rows = [json.loads(line) for line in (run / "eval-chat.jsonl").read_text().splitlines()]
  assert [(row["kind"], row["outcome"]) for row in rows] == [
    ("article", MISSED),
    ("ementa", MISSED),
  ]
  report = capsys.readouterr().out
  assert "eval-chat: run on 2 held-out questions" in report
  assert "answerable 2: correct 0, deferred 2, wrong shown 0" in report
  assert run_eval_chat(["--run", str(run), "--eval", str(held_out), "--search", "nada.db"]) == 1
