"""Chat tuning on synthetic conversations: turn format, answer-only mask, loss, JSONL."""

from pathlib import Path

import pytest
import torch
from torch.nn.functional import cross_entropy

from agavus.model.agathe import END_ID, EVAL_BATCH, AgatheConfig, start
from agavus.model.federal import PUBLIC_DOMAIN
from agavus.model.sft import (
  Conversation,
  TuneConfig,
  encode_conversation,
  load_conversations,
  masked_loss,
  prompt,
  render,
  rows,
  save_conversations,
  tune,
)
from agavus.model.tokenizer import END, decode, encode, train_tokenizer
from agavus.model.training import IGNORED

torch.set_num_threads(1)

QUESTION = "O que diz o art. 1º da Lei nº 99.999?"
ANSWER = "Art. 1º Esta Lei cria medidas fictícias.\n\nFonte: Lei nº 99.999, art. 1º."
FOLLOW_UP, SECOND = "E o art. 2º?", "Art. 2º Esta Lei entra em vigor."
CONVERSATION = Conversation(
  (("user", QUESTION), ("assistant", ANSWER), ("user", FOLLOW_UP), ("assistant", SECOND)),
  "rules:article",
  "https://example.gov.br/l99999.htm, art. 1",
  PUBLIC_DOMAIN,
)
TOKENIZER = train_tokenizer([QUESTION, ANSWER, FOLLOW_UP, SECOND] * 5, 300)
CONFIG = AgatheConfig(context=128, width=32, heads=4, kv_heads=2, layers=1)


def test_render_and_prompt_use_the_turn_format() -> None:
  pair = Conversation(CONVERSATION.turns[:2], "s", "ref", PUBLIC_DOMAIN)
  assert render(pair) == f"<|user|>{QUESTION}<|end|><|assistant|>{ANSWER}<|end|>"
  assert render(pair).startswith(prompt(QUESTION))
  assert prompt(QUESTION) == f"<|user|>{QUESTION}<|end|><|assistant|>"


def test_mask_marks_exactly_the_answers_and_their_end_tokens() -> None:
  ids, mask = encode_conversation(TOKENIZER, CONVERSATION)
  assert ids == encode(TOKENIZER, render(CONVERSATION))
  assert decode(TOKENIZER, [i for i, kept in zip(ids, mask, strict=True) if kept]) == (
    f"{ANSWER}{END}{SECOND}{END}"
  )
  assert decode(TOKENIZER, [i for i, kept in zip(ids, mask, strict=True) if not kept]) == (
    f"<|user|>{QUESTION}<|end|><|assistant|><|user|>{FOLLOW_UP}<|end|><|assistant|>"
  )


def test_rows_shift_targets_and_never_score_padding() -> None:
  short = Conversation((("user", "E?"), ("assistant", "Sim.")), "s", "ref", PUBLIC_DOMAIN)
  inputs, targets = rows(TOKENIZER, [CONVERSATION, short], CONFIG.context)
  ids, mask = encode_conversation(TOKENIZER, CONVERSATION)
  assert inputs[0].tolist() == [END_ID, *ids[:-1]]
  assert targets[0].tolist() == [i if kept else IGNORED for i, kept in zip(ids, mask, strict=True)]
  short_ids, _ = encode_conversation(TOKENIZER, short)
  assert (targets[1, len(short_ids) :] == IGNORED).all()
  assert (inputs[1, len(short_ids) + 1 :] == END_ID).all()


def test_masked_loss_scores_only_answer_tokens() -> None:
  model = start(TOKENIZER, CONFIG)
  ids, mask = encode_conversation(TOKENIZER, CONVERSATION)
  with torch.no_grad():
    scores = model.scores(torch.tensor([[END_ID, *ids[:-1]]]))[0]
  every = cross_entropy(scores, torch.tensor(ids), reduction="none")
  answers = every[torch.tensor(mask)].mean()
  assert masked_loss(model, TOKENIZER, [CONVERSATION]) == pytest.approx(float(answers), rel=1e-5)
  assert float(answers) != pytest.approx(float(every.mean()), rel=1e-3)


def test_masked_loss_is_the_mean_per_answer_token_across_blocks() -> None:
  model = start(TOKENIZER, CONFIG)
  short = Conversation((("user", "E?"), ("assistant", "Sim.")), "s", "ref", PUBLIC_DOMAIN)
  # One block of short answers, then one of long ones: a mean of block means would weigh
  # each short answer token as much as all the long ones together.
  shorts, longs = [short] * EVAL_BATCH, [CONVERSATION] * EVAL_BATCH
  counts = [
    int((rows(TOKENIZER, part, CONFIG.context)[1] != IGNORED).sum()) for part in (shorts, longs)
  ]
  means = [masked_loss(model, TOKENIZER, part) for part in (shorts, longs)]
  expected = (means[0] * counts[0] + means[1] * counts[1]) / sum(counts)
  assert masked_loss(model, TOKENIZER, shorts + longs) == pytest.approx(expected, rel=1e-5)
  assert expected != pytest.approx((means[0] + means[1]) / 2, rel=1e-3)


def test_tune_lowers_the_masked_loss_in_place() -> None:
  model = start(TOKENIZER, CONFIG)
  before = masked_loss(model, TOKENIZER, [CONVERSATION])
  settings = TuneConfig(rate=1e-2, batch=2, check_every=10)
  tuned, history = tune(model, [CONVERSATION], settings, steps=30, held_out=[CONVERSATION])
  assert tuned is model
  assert [step for step, _, _ in history] == [10, 20, 30]
  assert history[-1][2] < history[0][2] < before
  assert masked_loss(model, TOKENIZER, [CONVERSATION]) == pytest.approx(history[-1][2])


def test_conversations_need_known_roles_and_provenance() -> None:
  with pytest.raises(ValueError):
    Conversation((("system", "x"),), "s", "ref", PUBLIC_DOMAIN)
  with pytest.raises(ValueError):
    Conversation((("user", "x"),), "s", "", PUBLIC_DOMAIN)


def test_jsonl_round_trip_keeps_every_field(tmp_path: Path) -> None:
  path = tmp_path / "sft" / "rules.jsonl"
  save_conversations(path, [CONVERSATION, CONVERSATION])
  assert load_conversations(path) == [CONVERSATION, CONVERSATION]
  path.write_text('{"turns": [["user", 1]], "source": "s", "origin_ref": "r", "license": "l"}\n')
  with pytest.raises(ValueError, match=r"rules\.jsonl:1"):
    load_conversations(path)
