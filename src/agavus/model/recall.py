"""Recall probe: how much text a checkpoint reproduces verbatim (long-term memory).

Also the memorization test that `docs/llm-lab.md` asks for before a model trained on
DJEN text leaves the machine: give the start of a training text, see if the rest returns.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from agavus.model.agathe import AgatheModel
from agavus.model.bigram import BOUNDARY


@dataclass(frozen=True)
class RecallReport:
  """Share reproduced exactly, mean share matched before a mistake, loss per character."""

  texts: int
  exact: float
  matched: float
  loss: float


def _next_scores(model: AgatheModel, ids: Sequence[int]) -> torch.Tensor:
  """Scores for the token after `ids`, seeing only the most recent `context` of them."""
  recent = torch.tensor([list(ids[-model.config.context :])], device=model.config.device)
  return model.scores(recent)[0, -1]


def greedy(model: AgatheModel, prompt: str, chars: int) -> str:
  """Deterministic most likely continuation, stopping early at the boundary."""
  ids = model.vocab.encode(prompt)[:-1]
  written: list[int] = []
  with torch.no_grad():
    for _ in range(chars):
      chosen = int(_next_scores(model, ids).argmax())
      if chosen == BOUNDARY:
        break
      ids.append(chosen)
      written.append(chosen)
  return model.vocab.decode(written)


def _answer_loss(model: AgatheModel, prompt: str, answer: str) -> float:
  """Total -ln P of the answer tokens, each given everything before it (teacher-forced)."""
  ids = model.vocab.encode(prompt)[:-1]
  total = 0.0
  with torch.no_grad():
    for token in model.vocab.encode(answer)[1:-1]:
      total -= float(torch.log_softmax(_next_scores(model, ids), dim=-1)[token])
      ids.append(token)
  return total


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
