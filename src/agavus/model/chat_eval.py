"""Offline chat evaluation: greedy answers on held-out context questions (docs/llm-lab.md §9.6).

Each held-out conversation already holds the user turn the model sees (articles, then the
question). The tuned run answers it greedily; the answer goes through the same grounding
check as the chat server, with the shown articles read back whole from the search index.
Answerable questions end correct, wrongly answered or wrongly deferred; NOT_FOUND questions
end deferred or answered anyway. Wrong answers shown and answers shown on a negative are the
safety numbers and should be zero.
"""

import argparse
import json
import multiprocessing
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from agavus.config.settings import load_settings
from agavus.model.agathe import AgatheModel
from agavus.model.bundle import BUNDLE_FILE
from agavus.model.checkpoint import load_checkpoint
from agavus.model.generation import greedy_answer
from agavus.model.grounding import GROUNDED_SHARE, checked, citation, overlap
from agavus.model.law_search import SEARCH_FILE
from agavus.model.recall import sample_texts
from agavus.model.report import THREADS
from agavus.model.sft import SFT_DIR, Conversation, load_conversations, prompt
from agavus.model.sft_context import NOT_FOUND, question_of, shown_hits
from agavus.model.tokenizer import TOKENIZER_FILE

CORRECT = "correct"
WRONG = "wrong answer shown"
MISSED = "deferred, answerable"
DEFERRED = "deferred"
LEAKED = "answered a negative"
OUTCOMES = (CORRECT, MISSED, WRONG, DEFERRED, LEAKED)
# Loaded once per worker process by `_load`; a list so workers need no `global`.
_WORKER: list[AgatheModel] = []


@dataclass(frozen=True)
class ChatCase:
  """One held-out question: what the model wrote, what the chat would show, and the verdict."""

  source: str
  origin_ref: str
  question: str
  gold: str
  raw: str
  shown: str
  looped: bool
  outcome: str

  @property
  def kind(self) -> str:
    """'article', 'ementa', 'reverse' or 'teacher', plus '/negative' for NOT_FOUND questions."""
    origin, _, rest = self.source.partition(":")
    kind = "teacher" if origin == "teacher" else rest.partition(":")[0]
    return f"{kind}/negative" if self.gold == NOT_FOUND else kind

  @property
  def blocked(self) -> bool:
    """Whether the grounding check turned a written (non-empty) answer into the deferral."""
    written = self.raw.strip()
    return bool(written) and written != NOT_FOUND and self.shown == NOT_FOUND


def matches(shown: str, gold: str) -> bool:
  """Whether a shown answer gives the gold one: same citation, body copied from it."""
  return citation(shown) == citation(gold) and overlap(shown, [gold]) >= GROUNDED_SHARE


def outcome(shown: str, gold: str) -> str:
  """The verdict for a shown answer against the gold answer."""
  if gold == NOT_FOUND:
    return DEFERRED if shown == NOT_FOUND else LEAKED
  if shown == NOT_FOUND:
    return MISSED
  return CORRECT if matches(shown, gold) else WRONG


def judge(model: AgatheModel, conversation: Conversation, db: Path) -> ChatCase:
  """Answer one held-out context conversation greedily and judge what the chat would show."""
  (_, turn), (_, gold) = conversation.turns
  raw, looped = greedy_answer(model, prompt(turn))
  question = question_of(turn)
  shown = NOT_FOUND if looped else checked(raw, shown_hits(turn, db), question)
  return ChatCase(
    conversation.source,
    conversation.origin_ref,
    question,
    gold,
    raw,
    shown,
    looped,
    outcome(shown, gold),
  )


def _load(run: Path, bundle: Path, tokenizer: Path, data_dir: Path) -> None:
  torch.set_num_threads(THREADS)
  _WORKER.append(load_checkpoint(run, bundle, tokenizer, data_dir))


def _judged(job: tuple[Conversation, Path]) -> ChatCase:
  return judge(_WORKER[0], *job)


def summary(cases: Sequence[ChatCase]) -> list[str]:
  """Report lines: count per kind and outcome, then totals and the safety numbers."""
  counts = Counter((case.kind, case.outcome) for case in cases)
  lines = [f"{'kind':<18}" + "".join(f"{name:>22}" for name in OUTCOMES)]
  for kind in sorted({case.kind for case in cases}):
    lines.append(f"{kind:<18}" + "".join(f"{counts[kind, name]:>22}" for name in OUTCOMES))
  totals = Counter(case.outcome for case in cases)
  answerable = sum(totals[name] for name in (CORRECT, MISSED, WRONG))
  lines.append(
    f"answerable {answerable}: correct {totals[CORRECT]}, deferred {totals[MISSED]}, "
    f"wrong shown {totals[WRONG]}; negatives {totals[DEFERRED] + totals[LEAKED]}: "
    f"answered {totals[LEAKED]}"
  )
  blocked, looped = sum(case.blocked for case in cases), sum(case.looped for case in cases)
  lines.append(f"grounding check blocked {blocked} written answers ({looped} loops)")
  return lines


def run_eval_chat(arguments: Sequence[str]) -> int:
  """CLI: judge a chat-tuned run on held-out context questions; write every case as JSONL."""
  data_dir = load_settings().data_dir
  parser = argparse.ArgumentParser(prog="python -m agavus.model eval-chat")
  parser.add_argument("--run", type=Path, required=True, help="chat-tuned run folder")
  parser.add_argument("--eval", type=Path, default=data_dir / SFT_DIR / "context-eval.jsonl")
  parser.add_argument("--search", type=Path, default=data_dir / SEARCH_FILE)
  parser.add_argument("--tokenizer", type=Path, default=data_dir / TOKENIZER_FILE)
  parser.add_argument("--limit", type=int, help="seeded sample of the held-out questions")
  parser.add_argument("--workers", type=int, default=1, help=f"processes, {THREADS} threads each")
  parser.add_argument("--out", type=Path, help="cases JSONL (default: <run>/eval-chat.jsonl)")
  options = parser.parse_args(arguments)
  try:
    held_out = load_conversations(options.eval)
    picked = list(sample_texts(held_out, options.limit or len(held_out)))
    loading = (options.run, data_dir / BUNDLE_FILE, options.tokenizer, data_dir)
    if not options.search.is_file():
      raise ValueError(f"no search index at {options.search}")
    jobs = [(conversation, options.search) for conversation in picked]
    # Spawned, not forked: torch keeps threads that a fork would copy mid-flight.
    context = multiprocessing.get_context("spawn")
    workers = max(1, options.workers)
    with ProcessPoolExecutor(workers, context, initializer=_load, initargs=loading) as pool:
      cases = list(pool.map(_judged, jobs, chunksize=4))
  except (OSError, ValueError) as error:
    print(f"lab: eval-chat failed: {error}")
    return 1
  out: Path = options.out or options.run / "eval-chat.jsonl"
  rows = (json.dumps(asdict(case) | {"kind": case.kind}, ensure_ascii=False) for case in cases)
  out.write_text("".join(row + "\n" for row in rows), encoding="utf-8")
  print(f"eval-chat: {options.run.name} on {len(cases)} held-out questions → {out}")
  print("\n".join(summary(cases)))
  return 0
