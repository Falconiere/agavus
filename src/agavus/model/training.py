"""What every PyTorch lab model shares: starting weights, training, loss, sampling."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from agavus.model.data import Split

IGNORED = -100
CHECK_CHARS = 1_000_000
History = list[tuple[int, float, float]]
# Steps between progress lines when a run asks for them; checks and the last step print too.
LOG_EVERY = 100


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
  def log_every(self) -> int:
    """Steps between progress lines; 0 prints none."""
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


@dataclass(frozen=True)
class Progress:
  """Where a run stands: step of steps, the last batch's loss, wall seconds, latest check."""

  step: int
  steps: int
  loss: float
  seconds: float
  checked: tuple[float, float] | None = None

  def line(self) -> str:
    """One log line: 'step 1200/12000  batch 0.4120  0.180 s/step  eta 0h32m  …'."""
    per_step = self.seconds / self.step
    minutes = round(per_step * (self.steps - self.step) / 60)
    text = (
      f"step {self.step:>6}/{self.steps}  batch {self.loss:.4f}  {per_step:.3f} s/step"
      f"  eta {minutes // 60}h{minutes % 60:02d}m"
    )
    if self.checked is not None:
      text += f"  train {self.checked[0]:.4f}  held-out {self.checked[1]:.4f}"
    return text


def print_progress(progress: Progress) -> None:
  """Print the progress line at once, so a redirected log can be followed live."""
  print(progress.line(), flush=True)


def descend(
  weights: list[Tensor],
  mistake: Callable[[torch.Generator], Tensor],
  settings: Settings,
  *,
  steps: int,
  check: Callable[[], tuple[float, float]] | None,
) -> History:
  """Nudge `weights` `steps` times with AdamW; every `check_every` steps, record `check()`.

  With `log_every`, a progress line prints every `log_every` steps, at every check and at the
  last step; reading the batch loss waits for the device only then.
  """
  generator = torch.Generator().manual_seed(settings.seed)
  # Decay pulls matrices toward zero; gains and biases keep their scale.
  matrices = [weight for weight in weights if weight.dim() > 1]
  others = [weight for weight in weights if weight.dim() <= 1]
  groups = [
    {"params": matrices, "weight_decay": settings.weight_decay},
    {"params": others, "weight_decay": 0.0},
  ]
  optimizer = torch.optim.AdamW(groups, lr=settings.rate)
  history: History = []
  began = perf_counter()
  for number in range(1, steps + 1):
    loss = mistake(generator)
    slopes = torch.autograd.grad(loss, weights)
    for weight, slope in zip(weights, slopes, strict=True):
      weight.grad = slope
    optimizer.step()
    checked = None
    if check is not None and number % settings.check_every == 0:
      checked = check()
      history.append((number, *checked))
    every = settings.log_every
    if every and (checked or number % every == 0 or number == steps):
      print_progress(Progress(number, steps, float(loss.detach()), perf_counter() - began, checked))
  return history


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
  checked = check_sample(watch.train)

  def check() -> tuple[float, float]:
    return model.loss(checked), model.loss(watch.holdout)

  return descend(
    model.parameters(), mistake, settings, steps=steps, check=check if watch.holdout else None
  )


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
