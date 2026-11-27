"""Greedy answers as text: loops cut, the same answer every time."""

from dataclasses import replace

import torch

from agavus.model.agathe import END_ID, AgatheConfig, AgatheModel, start
from agavus.model.generation import LOOP_REPEATS, greedy_answer, looping
from agavus.model.tokenizer import encode, train_tokenizer

TOKENIZER = train_tokenizer(["um papel ção", "papel"] * 3, 300)
TINY = AgatheConfig(width=8, heads=2, kv_heads=1, context=64, layers=1)


def silent(context: int, favorite: int = END_ID) -> AgatheModel:
  """A model that always picks `favorite`; by default it answers nothing (the end token)."""
  model = start(TOKENIZER, replace(TINY, context=context))
  # With the residual paths silenced, only the token table decides.
  silenced = [weight for layer in model.layers for weight in (layer.mix, layer.down)]
  with torch.no_grad():
    for weight in silenced:
      weight.zero_()
    model.tokens.fill_(10.0)
    model.tokens[favorite].fill_(30.0)
  return model


def test_a_loop_is_one_short_cycle_repeated() -> None:
  assert looping([7] * LOOP_REPEATS) and looping([1, 2, 3] + [4, 5] * LOOP_REPEATS)
  assert not looping([7] * (LOOP_REPEATS - 1)) and not looping(list(range(40)))


def test_greedy_answers_repeat_and_stop_at_a_loop_or_the_end_token() -> None:
  model = silent(64, encode(TOKENIZER, "papel")[0])
  answer, looped = greedy_answer(model, "um")
  assert looped and answer.count("papel") == LOOP_REPEATS
  assert greedy_answer(model, "um") == (answer, looped)
  with torch.no_grad():
    model.tokens[END_ID].fill_(40.0)
  assert greedy_answer(model, "um") == ("", False)
