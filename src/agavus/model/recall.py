"""Recall probe: how much text a checkpoint reproduces verbatim (long-term memory).

Also the memorization test that `docs/llm-lab.md` asks for before a model trained on
DJEN text leaves the machine: give the start of a training text, see if the rest returns.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from agavus.model.agathe import END_ID, AgatheModel, framed
from agavus.model.tokenizer import decode, encode


@dataclass(frozen=True)
class RecallReport:
  """Share reproduced exactly, mean share matched before a mistake, loss per character."""

  texts: int
  exact: float
  matched: float
  loss: float


def greedy(model: AgatheModel, prompt: str, chars: int) -> str:
  """Deterministic most likely continuation of `chars` characters, stopping at the end token."""
  window = framed(model.tokenizer, prompt)[:-1][-model.config.context :]
  cache: list[tuple[torch.Tensor, torch.Tensor]] = []
  found = model.extend(window, 0, cache)
  written: list[int] = []
  # A token holds at least one byte, so `chars` UTF-8 characters need at most 4 per char.
  for position in range(len(window), len(window) + 4 * chars):
    chosen = int(found.argmax())
    if chosen == END_ID:
      break
    written.append(chosen)
    if len(decode(model.tokenizer, written)) >= chars:
      break
    found = model.extend([chosen], position, cache)
  return decode(model.tokenizer, written)[:chars]


def _answer_loss(model: AgatheModel, prompt: str, answer: str) -> float:
  """Total -ln P of the answer tokens, each given everything before it (teacher-forced)."""
  ids = framed(model.tokenizer, prompt)[:-1]
  answered = encode(model.tokenizer, answer)
  row = torch.tensor([(ids + answered)[-model.config.context - 1 :]], device=model.config.device)
  with torch.no_grad():
    chances = torch.log_softmax(model.scores(row[:, :-1])[0], dim=-1)
  told = row[0, 1:]
  # shortcut: one window, so an answer longer than the context only counts its last tokens.
  return -float(chances.gather(1, told[:, None])[max(0, len(told) - len(answered)) :].sum())


def _matched(expected: str, got: str) -> int:
  """Characters equal before the first mistake."""
  return next(
    (i for i, (a, b) in enumerate(zip(expected, got, strict=False)) if a != b),
    min(len(expected), len(got)),
  )


def recall(
  model: AgatheModel,
  texts: Sequence[str],
  *,
  prompt_chars: int = 64,
  answer_chars: int = 64,
) -> RecallReport:
  """Probe each text long enough: prompt with its start, compare the greedy answer."""
  exact = matched = loss = 0.0
  count = 0
  for text in texts:
    clean = " ".join(text.split())
    if len(clean) < prompt_chars + answer_chars:
      continue
    prompt = clean[:prompt_chars]
    answer = clean[prompt_chars : prompt_chars + answer_chars]
    got = greedy(model, prompt, answer_chars)
    count += 1
    exact += got == answer
    matched += _matched(answer, got) / answer_chars
    loss += _answer_loss(model, prompt, answer) / answer_chars
  if count == 0:
    return RecallReport(0, 0.0, 0.0, 0.0)
  return RecallReport(count, exact / count, matched / count, loss / count)


def sample_texts(texts: Sequence[str], limit: int, seed: int = 0) -> tuple[str, ...]:
  """Seeded sample without replacement; everything if there are fewer than `limit`."""
  if len(texts) <= limit:
    return tuple(texts)
  return tuple(random.Random(seed).sample(list(texts), limit))
