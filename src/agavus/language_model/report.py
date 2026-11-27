"""What every lab experiment prints: chosen steps, losses, a sample."""

from collections.abc import Sequence
from typing import Protocol

from agavus.language_model.data import Split
from agavus.language_model.training import History

SAMPLE_SEED = 7
SAMPLE_CHARS = 200
# More threads only fight over work this small.
THREADS = 4


class Scored(Protocol):
  """A trained model that can be examined and can write."""

  def loss(self, texts: Sequence[str]) -> float:
    """Mean surprise per token."""
    ...

  def sample(self, *, seed: int, max_chars: int) -> str:
    """Seeded text, one token at a time."""
    ...


def chosen_steps(history: History, shown: int) -> int:
  """Print some checks; the step with the lowest validation loss wins."""
  for number, training, validation in history[:: max(len(history) // shown, 1)]:
    print(f"step {number:>5}  training {training:.3f}  validation {validation:.3f}")
  best = min(history, key=lambda entry: entry[2])[0]
  print(f"steps chosen on validation: {best}")
  return best


def report(model: Scored, split: Split) -> tuple[float, float, str]:
  """Print and return training loss, exam loss and a seeded sample."""
  training, holdout = model.loss(split.train), model.loss(split.holdout)
  sample = model.sample(seed=SAMPLE_SEED, max_chars=SAMPLE_CHARS)
  print(f"loss train:    {training:.4f}")
  print(f"loss holdout:  {holdout:.4f}")
  print(f"sample: {sample}")
  return training, holdout, sample
