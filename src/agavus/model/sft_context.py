"""Chat-tuning conversations that answer from retrieved articles placed in the user turn (§9).

Each question is searched; the cited article joins the hits, so the answer is in context.
A seeded share of conversations has the cited article taken out and must defer instead.
"""

import random
from collections.abc import Sequence
from pathlib import Path

from agavus.model.law_search import Hit, article, search
from agavus.model.sft import Conversation
from agavus.model.sft_rules import law_of

CONTEXT_CHARS = 1200
SEARCHED = 3
NOT_FOUND = "Os textos encontrados não respondem a essa pergunta."
PAIR = ("user", "assistant")


def _clipped(text: str) -> str:
  """At most CONTEXT_CHARS characters, marked when cut."""
  return text if len(text) <= CONTEXT_CHARS else text[:CONTEXT_CHARS].rstrip() + "…"


def with_context(question: str, hits: Sequence[Hit]) -> str:
  """The user turn: numbered articles, then the question. The chat sends the same text."""
  shown = "".join(
    f"[{index}] {hit.name}, art. {hit.number}:\n{_clipped(hit.text)}\n\n"
    for index, hit in enumerate(hits, 1)
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


def contextualize(
  conversations: Sequence[Conversation], db: Path, *, seed: int, negatives: float = 0.1
) -> list[Conversation]:
  """Question-answer pairs with their search hits in the user turn; provenance kept.

  Conversations that are not a single pair, or whose cited article is not indexed, are
  skipped. A `negatives` share answers NOT_FOUND, with the cited article absent.
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
    negative = chooser.random() < negatives
    hits = _shown(search(db, question, SEARCHED + 1), own, negative, chooser)
    turns = (
      ("user", with_context(question, hits)),
      ("assistant", NOT_FOUND if negative else answer),
    )
    found.append(
      Conversation(
        turns, f"{conversation.source}:context", conversation.origin_ref, conversation.license
      )
    )
  return found


def teacher_for_training(
  teacher: Sequence[Conversation], held_out: Sequence[Conversation]
) -> list[Conversation]:
  """Teacher conversations whose law has no held-out conversation."""
  held_laws = {law_of(conversation) for conversation in held_out}
  return [conversation for conversation in teacher if law_of(conversation) not in held_laws]
