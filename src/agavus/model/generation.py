"""Decoding Agathe answers as text: partial characters held back, loops cut (docs/llm-lab.md §9).

The chat server streams these pieces; `eval-chat` joins them. Both decode chat answers
greedily, so the same question gives the same answer.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from secrets import randbits
from time import perf_counter

from agavus.model.agathe import AgatheModel
from agavus.model.tokenizer import decode

PARTIAL = "\ufffd"
# Chat answers quote whole articles; plain continuations stay sample-sized.
CHAT_CHARS = 2000
# A tail of the same 1..LOOP_PERIOD tokens LOOP_REPEATS times running is a loop, not an answer.
LOOP_PERIOD = 8
LOOP_REPEATS = 6


def looping(ids: Sequence[int]) -> bool:
  """Whether the ids end in one short cycle repeated LOOP_REPEATS times."""
  for period in range(1, LOOP_PERIOD + 1):
    span = period * LOOP_REPEATS
    if len(ids) >= span and all(ids[-i] == ids[-i - period] for i in range(1, span - period + 1)):
      return True
  return False


# Mutable on purpose: `pieces` fills it while the caller consumes the text.
@dataclass
class Tally:
  """What one generation spent: decoding seconds and whether it ended in a loop."""

  seconds: float = 0.0
  looped: bool = False


def pieces(
  model: AgatheModel, prefix: str, max_chars: int, greedy: bool, tally: Tally
) -> Iterator[str]:
  """New text as it decodes, at most `max_chars`; stops at the end token or a loop."""
  written: list[int] = []
  sent = ""
  tokens = model.continuation(prefix, seed=randbits(63), max_tokens=max_chars, greedy=greedy)
  while True:
    began = perf_counter()
    token = next(tokens, None)
    tally.seconds += perf_counter() - began
    if token is not None:
      written.append(token)
      tally.looped = looping(written)
    text = decode(model.tokenizer, written)[:max_chars]
    # A character split across tokens waits for its last byte.
    if text.endswith(PARTIAL) and token is not None:
      continue
    if len(text) > len(sent):
      yield text[len(sent) :]
      sent = text
    if token is None or tally.looped or len(sent) == max_chars:
      return


def greedy_answer(model: AgatheModel, prefix: str, max_chars: int = CHAT_CHARS) -> tuple[str, bool]:
  """The whole greedy answer after `prefix`, and whether it was cut by a loop."""
  tally = Tally()
  answer = "".join(pieces(model, prefix, max_chars, True, tally))
  return answer, tally.looped
