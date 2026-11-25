"""Run the lab: `uv run python -m agavus.lab [bigram|network|neural]`."""

import sys

import torch

from agavus.config.settings import load_settings
from agavus.lab.bigram import BigramModel, build_vocab, train_bigram, uniform_loss
from agavus.lab.data import Split, load_articles, split_articles
from agavus.lab.network import CLUES, REFUSED, TEXTS, forward, start, train, train_single_neuron
from agavus.lab.neural import NeuralConfig, NeuralModel
from agavus.lab.neural import train as train_neural

SAMPLE_SEED = 7
SAMPLE_CHARS = 200
HIDDEN = 8
NETWORK_SEED = 0
STEPS = 2000
# More threads only fight over work this small.
THREADS = 4
NEURAL_STEPS = 4000
SHOWN_CHECKS = 8
RATE = 0.5
SHOWN_STEPS = (0, 10, 50, 100, 200, 500, 1000, 2000)


def _statutes() -> Split | None:
  articles = load_articles(load_settings().data_dir)
  if not articles:
    print("lab: no statute articles — ingest statutes first")
    return None
  return split_articles(articles)


def _report(model: BigramModel | NeuralModel, split: Split) -> None:
  print(f"loss train:    {model.loss(split.train):.4f}")
  print(f"loss holdout:  {model.loss(split.holdout):.4f}")
  print(f"sample: {model.sample(seed=SAMPLE_SEED, max_chars=SAMPLE_CHARS)}")


def run_bigram() -> int:
  """Train the bigram on statutes and report losses and a sample."""
  split = _statutes()
  if split is None:
    return 1
  model = train_bigram(split.train)
  print(f"articles: {len(split.train)} train / {len(split.holdout)} holdout")
  print(f"vocabulary: {model.vocab.size} tokens")
  print(f"loss uniform:  {uniform_loss(model.vocab):.4f}")
  _report(model, split)
  return 0


def run_network() -> int:
  """Watch the tiny network learn that two negatives cancel."""
  network, losses = train(start(HIDDEN, NETWORK_SEED), CLUES, REFUSED, steps=STEPS, rate=RATE)
  for number in SHOWN_STEPS:
    print(f"step {number:>5}  loss {losses[number]:.3f}")
  alone = train_single_neuron(CLUES, REFUSED, steps=STEPS, rate=RATE)
  guesses = forward(network, CLUES).guess
  print("refused?        answer  one neuron  network")
  for text, answer, single, guess in zip(TEXTS, REFUSED, alone, guesses, strict=True):
    print(f"{text:<15} {answer:>6.0f}  {single:>10.2f}  {guess:>7.2f}")
  return 0


def run_neural() -> int:
  """Neural model: steps chosen on validation, exam opened once at the end."""
  split = _statutes()
  if split is None:
    return 1
  inner = split_articles(split.train)
  torch.set_num_threads(THREADS)
  vocab = build_vocab(split.train)
  config = NeuralConfig()
  _, history = train_neural(
    inner.train, vocab, config, steps=NEURAL_STEPS, validation=inner.holdout
  )
  for number, training, validation in history[:: len(history) // SHOWN_CHECKS]:
    print(f"step {number:>5}  training {training:.3f}  validation {validation:.3f}")
  best = min(history, key=lambda entry: entry[2])[0]
  model, _ = train_neural(split.train, vocab, config, steps=best)
  bigram = train_bigram(split.train)
  print(f"steps chosen on validation: {best}")
  print(f"loss bigram holdout: {bigram.loss(split.holdout):.4f}")
  _report(model, split)
  return 0


def main(arguments: list[str]) -> int:
  """Choose the experiment; the bigram is the default."""
  experiments = {"bigram": run_bigram, "network": run_network, "neural": run_neural}
  name = arguments[0] if arguments else "bigram"
  if name not in experiments:
    print(f"lab: unknown experiment {name!r}; choose from {', '.join(experiments)}")
    return 1
  return experiments[name]()


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
