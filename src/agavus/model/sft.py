"""Chat tuning (SFT) for Agathe: conversations in turn format, loss on answers only.

A conversation is rendered as `<|user|>question<|end|><|assistant|>answer<|end|>` per pair
(docs/llm-lab.md §9.5). Tuning scores only the assistant's answer tokens and their closing
end token, so the model learns to answer, not to write questions.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.nn.functional import cross_entropy

from agavus.model.agathe import END_ID, EVAL_BATCH, AgatheModel
from agavus.model.federal import save_bytes
from agavus.model.tokenizer import ASSISTANT, END, USER, encode
from agavus.model.training import IGNORED, History, descend, mean_surprise

ROLES = {"user": USER, "assistant": ASSISTANT}
# The training loss is checked on at most this many conversations, and so is held-out.
CHECK_CONVERSATIONS = 256


SFT_DIR = Path("lab") / "sft"


@dataclass(frozen=True)
class Conversation:
  """Turns as (role, text), with where they came from and under which license."""

  turns: tuple[tuple[str, str], ...]
  source: str
  origin_ref: str
  license: str

  def __post_init__(self) -> None:
    """Refuse unknown roles and conversations without provenance."""
    if any(role not in ROLES for role, _ in self.turns):
      raise ValueError(f"roles must be one of {sorted(ROLES)}")
    if not self.origin_ref or not self.license:
      raise ValueError("every conversation needs an origin_ref and a license")


@dataclass(frozen=True)
class TuneConfig:
  """How chat tuning trains: a lower rate than pretraining keeps what the model knows."""

  rate: float = 1e-4
  weight_decay: float = 0.1
  batch: int = 16
  check_every: int = 100
  seed: int = 0


def render_turns(turns: Sequence[tuple[str, str]]) -> str:
  """Each turn as its role token, its text and an end token."""
  return "".join(f"{ROLES[role]}{text}{END}" for role, text in turns)


def render(conversation: Conversation) -> str:
  """The conversation in turn format."""
  return render_turns(conversation.turns)


def prompt(user_text: str) -> str:
  """What the chat sends for a question: the user turn, then the assistant's role token."""
  return f"{USER}{user_text}{END}{ASSISTANT}"


def encode_conversation(
  tokenizer: Tokenizer, conversation: Conversation
) -> tuple[list[int], list[bool]]:
  """Ids of the rendered text, and True exactly on answer tokens and each answer's end."""
  ids: list[int] = []
  mask: list[bool] = []
  for role, text in conversation.turns:
    answer = role == "assistant"
    for piece, scored in ((ROLES[role], False), (text, answer), (END, answer)):
      found = encode(tokenizer, piece)
      ids.extend(found)
      mask.extend([scored] * len(found))
  return ids, mask


def rows(
  tokenizer: Tokenizer, conversations: Sequence[Conversation], context: int
) -> tuple[Tensor, Tensor]:
  """Inputs and next-token targets, one row each, IGNORED wherever the next token isn't masked.

  Each row starts with the end token, as the chat's window does; long conversations are cut
  at `context`, and rows are padded to the longest with end tokens that are never scored.
  """
  cut = []
  for conversation in conversations:
    ids, mask = encode_conversation(tokenizer, conversation)
    cut.append(([END_ID, *ids][: context + 1], [False, *mask][: context + 1]))
  width = max((len(ids) for ids, _ in cut), default=1)
  inputs: list[list[int]] = []
  targets: list[list[int]] = []
  for ids, mask in cut:
    answers = [token if scored else IGNORED for token, scored in zip(ids, mask, strict=True)]
    padding = width - len(ids)
    inputs.append(ids[:-1] + [END_ID] * padding)
    targets.append(answers[1:] + [IGNORED] * padding)
  return torch.tensor(inputs), torch.tensor(targets)


def masked_loss(
  model: AgatheModel, tokenizer: Tokenizer, conversations: Sequence[Conversation]
) -> float:
  """Mean nats per answer token over `conversations`."""
  total, count = 0.0, 0
  device = model.config.device
  for start in range(0, len(conversations), EVAL_BATCH):
    inputs, targets = rows(
      tokenizer, conversations[start : start + EVAL_BATCH], model.config.context
    )
    scored = int((targets != IGNORED).sum())
    if scored:
      scored = model.eval_rows()
      mean = mean_surprise(lambda ids: model.scores(ids.to(device)), inputs, targets, scored)
      total, count = total + mean * scored, count + scored
  return total / max(1, count)


def _sample(conversations: Sequence[Conversation]) -> Sequence[Conversation]:
  """Evenly spaced conversations, at most CHECK_CONVERSATIONS of them."""
  return conversations[:: max(1, -(-len(conversations) // CHECK_CONVERSATIONS))]


def tune(
  model: AgatheModel,
  conversations: Sequence[Conversation],
  settings: TuneConfig,
  *,
  steps: int,
  held_out: Sequence[Conversation],
) -> tuple[AgatheModel, History]:
  """Fine-tune `model` in place on answers; history is (step, train, held-out) masked loss.

  Conversations are encoded with the model's own tokenizer.
  """
  if not conversations:
    raise ValueError("no conversations to tune on")
  tokenizer, device, context = model.tokenizer, model.config.device, model.config.context
  # Dropout draws from the global generator.
  torch.manual_seed(settings.seed)

  def mistake(generator: torch.Generator) -> Tensor:
    picks = torch.randint(len(conversations), (settings.batch,), generator=generator).tolist()
    inputs, targets = rows(tokenizer, [conversations[pick] for pick in picks], context)
    scores = model.learning_scores(inputs.to(device))
    targets = targets.to(device).flatten()
    total = cross_entropy(scores.flatten(0, 1), targets, ignore_index=IGNORED, reduction="sum")
    # A batch without answer tokens contributes nothing instead of dividing by zero.
    return total / max(1, int((targets != IGNORED).sum()))

  fixed, watched = _sample(conversations), _sample(held_out)

  def check() -> tuple[float, float]:
    return masked_loss(model, tokenizer, fixed), masked_loss(model, tokenizer, watched)

  history = descend(
    model.parameters(), mistake, settings, steps=steps, check=check if held_out else None
  )
  return model, history


def save_conversations(path: Path, conversations: Sequence[Conversation]) -> None:
  """One JSON object per line with every field, written whole or not at all."""
  lines = [
    json.dumps(
      {
        "turns": [list(turn) for turn in conversation.turns],
        "source": conversation.source,
        "origin_ref": conversation.origin_ref,
        "license": conversation.license,
      },
      ensure_ascii=False,
    )
    + "\n"
    for conversation in conversations
  ]
  save_bytes(path, "".join(lines).encode())


def _text(row: dict[str, object], key: str) -> str:
  """A string field, refused otherwise."""
  found = row[key]
  if not isinstance(found, str):
    raise ValueError(f"{key} must be text")
  return found


def load_conversations(path: Path) -> list[Conversation]:
  """Conversations saved by `save_conversations`; malformed lines raise ValueError."""
  found: list[Conversation] = []
  for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line:
      continue
    try:
      row = json.loads(line)
      turns = tuple((role, text) for role, text in row["turns"])
      if not all(isinstance(part, str) for turn in turns for part in turn):
        raise ValueError("turns must be (role, text) text pairs")
      fields = (_text(row, key) for key in ("source", "origin_ref", "license"))
      found.append(Conversation(turns, *fields))
    except (KeyError, TypeError, ValueError) as error:
      raise ValueError(f"{path}:{number}: {error}") from error
  return found
