"""Chat-tuning conversations that answer from retrieved articles placed in the user turn (§9).

Each question is searched; the cited article (or ementa) joins the hits, so the answer is in
context. Seeded shares must defer instead: one with the cited text taken out, one shown the
articles found for another law's question.
"""

import multiprocessing
import random
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from agavus.model.law_search import EMENTA, Hit, article, search
from agavus.model.sft import Conversation
from agavus.model.sft_rules import law_of

CONTEXT_CHARS = 1200
SEARCHED = 3
NOT_FOUND = "Os textos encontrados não respondem a essa pergunta."
PAIR = ("user", "assistant")
# Shard i of a parallel build draws from seed + SHARD_SEEDS * i, so one worker equals serial.
SHARD_SEEDS = 1000
# Off-topic deferrals the build adds by default: the chat must also defer when nothing fits.
UNRELATED = 0.05


@dataclass(frozen=True)
class Deferrals:
  """Shares of conversations that answer NOT_FOUND: cited text removed, or off-topic hits."""

  negatives: float = 0.1
  unrelated: float = 0.0


def _clipped(text: str) -> str:
  """At most CONTEXT_CHARS characters, marked when cut."""
  return text if len(text) <= CONTEXT_CHARS else text[:CONTEXT_CHARS].rstrip() + "…"


def cited(hit: Hit) -> str:
  """How a hit is labeled: '<law>, art. N', or '<law>, ementa'."""
  return f"{hit.name}, ementa" if hit.number == EMENTA else f"{hit.name}, art. {hit.number}"


def with_context(question: str, hits: Sequence[Hit]) -> str:
  """The user turn: numbered articles, then the question. The chat sends the same text."""
  shown = "".join(
    f"[{index}] {cited(hit)}:\n{_clipped(hit.text)}\n\n" for index, hit in enumerate(hits, 1)
  )
  return f"{shown}Pergunta: {question}"


def _shown(hits: list[Hit], own: Hit, negative: bool, chooser: random.Random) -> list[Hit]:
  """SEARCHED hits: without `own` for a negative, else with it at a seeded place if missing."""
  others = [hit for hit in hits if hit.origin_ref != own.origin_ref]
  if negative:
    return others[:SEARCHED]
  shown = hits[:SEARCHED]
  if own.origin_ref in {hit.origin_ref for hit in shown}:
    return shown
  kept = others[: SEARCHED - 1]
  kept.insert(chooser.randint(0, len(kept)), own)
  return kept


def _unrelated(
  db: Path, conversations: Sequence[Conversation], own: Conversation, chooser: random.Random
) -> list[Hit]:
  """The hits for a seeded other conversation's question, none from `own`'s law."""
  other = conversations[chooser.randrange(len(conversations))]
  hits = search(db, other.turns[0][1], SEARCHED * 2)
  return [hit for hit in hits if hit.origin_ref.partition(", ")[0] != law_of(own)][:SEARCHED]


def contextualize(
  conversations: Sequence[Conversation],
  db: Path,
  *,
  seed: int,
  negatives: float = 0.1,
  unrelated: float = 0.0,
) -> list[Conversation]:
  """Question-answer pairs with their search hits in the user turn; provenance kept.

  Conversations that are not a single pair, or whose cited text is not indexed, are skipped.
  A `negatives` share answers NOT_FOUND with the cited text absent; an `unrelated` share
  answers NOT_FOUND with the hits of another conversation's question.
  """
  chooser = random.Random(seed)
  found = []
  for conversation in conversations:
    if tuple(role for role, _ in conversation.turns) != PAIR:
      continue
    own = article(db, conversation.origin_ref)
    if own is None:
      continue
    (_, question), (_, answer) = conversation.turns
    draw = chooser.random()
    if draw < negatives:
      hits = _shown(search(db, question, SEARCHED + 1), own, True, chooser)
    elif draw < negatives + unrelated:
      hits = _unrelated(db, conversations, conversation, chooser)
    else:
      hits = _shown(search(db, question, SEARCHED + 1), own, False, chooser)
    turns = (
      ("user", with_context(question, hits)),
      ("assistant", answer if draw >= negatives + unrelated else NOT_FOUND),
    )
    found.append(
      Conversation(
        turns, f"{conversation.source}:context", conversation.origin_ref, conversation.license
      )
    )
  return found


def _shard(
  job: tuple[int, Sequence[Conversation]], db: Path, shares: Deferrals
) -> list[Conversation]:
  seed, part = job
  return contextualize(part, db, seed=seed, negatives=shares.negatives, unrelated=shares.unrelated)


def contextualize_parallel(
  conversations: Sequence[Conversation],
  db: Path,
  *,
  seed: int,
  workers: int,
  shares: Deferrals = Deferrals(),
) -> list[Conversation]:
  """`contextualize` split into `workers` interleaved shards, each searched in its process.

  Search is CPU-bound (~0.2 s per question), so shards cut the build by the worker count.
  Shard i is seeded seed + SHARD_SEEDS * i; one worker gives the serial result.
  """
  if workers <= 1:
    return _shard((seed, conversations), db, shares)
  jobs = [(seed + SHARD_SEEDS * i, conversations[i::workers]) for i in range(workers)]
  # Spawned, not forked: torch keeps threads that a fork would copy mid-flight.
  with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
    parts = pool.map(partial(_shard, db=db, shares=shares), jobs)
    return [conversation for part in parts for conversation in part]


def teacher_for_training(
  teacher: Sequence[Conversation], held_out: Sequence[Conversation]
) -> list[Conversation]:
  """Teacher conversations whose law has no held-out conversation."""
  held_laws = {law_of(conversation) for conversation in held_out}
  return [conversation for conversation in teacher if law_of(conversation) not in held_laws]
