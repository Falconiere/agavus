"""Recall probe on a tiny model that memorizes one synthetic text."""

import torch

from agavus.model.agathe import END_ID, AgatheConfig, start, train
from agavus.model.data import Split
from agavus.model.recall import greedy, recall, sample_texts
from agavus.model.tokenizer import train_tokenizer

torch.set_num_threads(1)

TEXT = "a guarda do filho sera compartilhada entre os pais sempre que possivel no caso"
PROMPT, ANSWER = 16, 16
LIMIT = 5
CONFIG = AgatheConfig(context=32, width=32, heads=2, layers=1, batch=8, rate=1e-2)
TOKENIZER = train_tokenizer([TEXT] * 3, 300)


def _memorized() -> tuple[float, float, float]:
  model, _ = train(Split((TEXT,) * 4, ()), TOKENIZER, CONFIG, steps=150)
  trained = recall(model, [TEXT], prompt_chars=PROMPT, answer_chars=ANSWER)
  blank = recall(start(TOKENIZER, CONFIG), [TEXT], prompt_chars=PROMPT, answer_chars=ANSWER)
  return trained.exact, trained.loss, blank.loss


def test_memorized_text_is_recalled() -> None:
  exact, trained_loss, blank_loss = _memorized()
  assert exact == 1.0
  assert trained_loss < blank_loss


def test_short_texts_are_skipped() -> None:
  model = start(TOKENIZER, CONFIG)
  report = recall(model, ["short", TEXT[:20]], prompt_chars=PROMPT, answer_chars=ANSWER)
  assert (report.texts, report.exact, report.matched, report.loss) == (0, 0.0, 0.0, 0.0)


def test_sample_is_seeded_and_capped() -> None:
  texts = tuple(f"text {n}" for n in range(20))
  first = sample_texts(texts, LIMIT, seed=1)
  assert first == sample_texts(texts, LIMIT, seed=1)
  assert len(set(first)) == LIMIT and set(first) <= set(texts)
  assert sample_texts(texts, 50) == texts


def test_greedy_stops_at_boundary() -> None:
  model = start(TOKENIZER, CONFIG)
  # Make the end token the most likely next token: the output reads the token table.
  model.tokens.data[END_ID] = 0.0
  assert greedy(model, "a guarda", 10) != ""
  model.layers[0].down.data.zero_()
  model.layers[0].mix.data.zero_()
  model.tokens.data[:] = model.tokens.data[:, :1].abs() + 1.0
  model.tokens.data[END_ID] = 10.0
  assert greedy(model, "a guarda", 10) == ""
