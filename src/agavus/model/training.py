"""What every PyTorch lab model shares: starting weights, training, loss, sampling."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from agavus.model.data import Split

IGNORED = -100
CHECK_CHARS = 1_000_000
History = list[tuple[int, float, float]]


def check_sample(texts: Sequence[str], chars: int = CHECK_CHARS) -> tuple[str, ...]:
  """Evenly spaced texts totalling about `chars`, so each training-loss check stays cheap."""
  stride = max(1, -(-sum(map(len, texts)) // chars))
  return tuple(texts[::stride])


def synchronize(device: str) -> None:
  """Wait for queued accelerator work at a timing boundary."""
  kind = torch.device(device).type
  if kind == "mps":
    torch.mps.synchronize()
  elif kind == "cuda":
    torch.cuda.synchronize(device)


class Settings(Protocol):
  """How a model trains; both lab configs have these fields."""

  @property
  def rate(self) -> float:
    """How big each nudge is."""
    ...

  @property
  def weight_decay(self) -> float:
    """How strongly weights are pulled back toward zero."""
    ...

  @property
  def check_every(self) -> int:
    """Steps between loss checks."""
    ...

  @property
  def seed(self) -> int:
    """Makes every random choice repeatable."""
    ...


class Trainable(Protocol):
  """A model training can adjust and score."""

  def parameters(self) -> list[Tensor]:
    """Everything that training adjusts."""
    ...

  def loss(self, texts: Sequence[str]) -> float:
    """Mean surprise per token."""
    ...


@dataclass(frozen=True)
class Init:
  """Random starting weights, reproducible by seed."""

  generator: torch.Generator
  device: str

  @classmethod
  def seeded(cls, seed: int, device: str) -> "Init":
    """A fresh generator for this seed."""
    return cls(torch.Generator().manual_seed(seed), device)

  def matrix(self, rows: int, columns: int, size: float) -> Tensor:
    """Random numbers around zero, about `size` big."""
    values = torch.randn(rows, columns, generator=self.generator) * size
    return values.to(self.device).requires_grad_()

  def zeros(self, length: int) -> Tensor:
    """Biases start at zero."""
    return torch.zeros(length, device=self.device).requires_grad_()


def fit(
  model: Trainable,
  mistake: Callable[[torch.Generator], Tensor],
  settings: Settings,
  *,
  steps: int,
  watch: Split,
) -> History:
  """Nudge the weights `steps` times.

  When `watch.holdout` has texts, history holds (step, training loss, validation loss).
  """
  generator = torch.Generator().manual_seed(settings.seed)
  weights = model.parameters()
  optimizer = torch.optim.AdamW(weights, lr=settings.rate, weight_decay=settings.weight_decay)
  history: History = []
  checked = check_sample(watch.train)
  for number in range(1, steps + 1):
    slopes = torch.autograd.grad(mistake(generator), weights)
    for weight, slope in zip(weights, slopes, strict=True):
      weight.grad = slope
    optimizer.step()
    if watch.holdout and number % settings.check_every == 0:
      history.append((number, model.loss(checked), model.loss(watch.holdout)))
  return history


def mean_surprise(
  scores: Callable[[Tensor], Tensor], inputs: Tensor, answers: Tensor, batch: int
) -> float:
  """Mean -ln P(true token) over every answer that isn't IGNORED."""
  total = 0.0
  with torch.no_grad():
    for start in range(0, len(answers), batch):
      part = slice(start, start + batch)
      found = scores(inputs[part])
      total += float(
        cross_entropy(
          found.reshape(-1, found.shape[-1]),
          answers[part].to(found.device).flatten(),
          ignore_index=IGNORED,
          reduction="sum",
        )
      )
  return total / int((answers != IGNORED).sum())


def pick(scores: Tensor, generator: torch.Generator) -> int:
  """Draw the next token from the chances the scores give."""
  chances = torch.softmax(scores.detach(), dim=0).cpu()
  return int(torch.multinomial(chances, 1, generator=generator).item())
