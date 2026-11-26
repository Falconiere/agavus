"""Run the lab: `uv run python -m agavus.lab [bigram|network|neural|bpe|gpt]`."""

import sys

import torch

from agavus.config.settings import load_settings
from agavus.lab.bigram import (
  BigramModel,
  CharVocab,
  build_vocab,
  loss_per_character,
  train_bigram,
  uniform_loss,
)
from agavus.lab.bpe import FIRST_MERGE, BpeVocab, train_bpe
from agavus.lab.data import Split, load_articles, split_articles
from agavus.lab.gpt import GptConfig, GptModel
from agavus.lab.gpt import train as train_gpt
from agavus.lab.network import CLUES, REFUSED, TEXTS, forward, start, train, train_single_neuron
from agavus.lab.neural import NeuralConfig, NeuralModel
from agavus.lab.neural import train as train_neural
from agavus.lab.training import History

SAMPLE_SEED = 7
SAMPLE_CHARS = 200
HIDDEN = 8
NETWORK_SEED = 0
STEPS = 2000
# More threads only fight over work this small.
THREADS = 4
NEURAL_STEPS = 4000
SHOWN_CHECKS = 8
# Chosen on validation articles (learning/14-word-pieces.md).
BIGRAM_MERGES = 1000
NEURAL_MERGES = 100
BPE_NEURAL_STEPS = 3000
SHOWN_PIECES = 20
EXAMPLE_WORDS = ("guarda", "guardar", "medida protetiva", "descumprimento")
GPT_STEPS = 6000
GPT_CHECK_EVERY = 500
ATTENTION_TEXT = "Art. 22. Constatada a prática de violência doméstica"
SHOWN_LOOKS = 3
RATE = 0.5
SHOWN_STEPS = (0, 10, 50, 100, 200, 500, 1000, 2000)


def _statutes() -> Split | None:
  articles = load_articles(load_settings().data_dir)
  if not articles:
    print("lab: no statute articles — ingest statutes first")
    return None
  return split_articles(articles)


def _report(model: BigramModel | NeuralModel | GptModel, split: Split) -> None:
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


def _chosen_steps(history: History, shown: int) -> int:
  """Print some checks; the step with the lowest validation loss wins."""
  for number, training, validation in history[:: max(len(history) // shown, 1)]:
    print(f"step {number:>5}  training {training:.3f}  validation {validation:.3f}")
  best = min(history, key=lambda entry: entry[2])[0]
  print(f"steps chosen on validation: {best}")
  return best


def _studied() -> tuple[Split, CharVocab] | None:
  """Statutes split for the exam, with a letter vocabulary from the study side."""
  split = _statutes()
  if split is None:
    return None
  torch.set_num_threads(THREADS)
  return split, build_vocab(split.train)


def run_neural() -> int:
  """Neural model: steps chosen on validation, exam opened once at the end."""
  studied = _studied()
  if studied is None:
    return 1
  split, vocab = studied
  config = NeuralConfig()
  _, history = train_neural(split_articles(split.train), vocab, config, steps=NEURAL_STEPS)
  best = _chosen_steps(history, SHOWN_CHECKS)
  model, _ = train_neural(Split(split.train, ()), vocab, config, steps=best)
  print(f"loss bigram holdout: {train_bigram(split.train).loss(split.holdout):.4f}")
  _report(model, split)
  return 0


def run_bpe() -> int:
  """Word pieces: what they look like, and the exam per character."""
  split = _statutes()
  if split is None:
    return 1
  torch.set_num_threads(THREADS)
  full = train_bpe(split.train, BIGRAM_MERGES)
  print("first pieces:", [full.piece(FIRST_MERGE + rank) for rank in range(SHOWN_PIECES)])
  for word in EXAMPLE_WORDS:
    print(f"{word}: {[full.piece(token) for token in full.encode(word)[1:-1]]}")
  bigram = train_bigram(split.train, vocab=full)
  small = BpeVocab(full.merges[:NEURAL_MERGES])
  neural, _ = train_neural(Split(split.train, ()), small, NeuralConfig(), steps=BPE_NEURAL_STEPS)
  for name, model in (("bigram", bigram), ("neural", neural)):
    loss = loss_per_character(model.loss(split.holdout), split.holdout, model.vocab)
    print(f"{name} + BPE holdout, per character: {loss:.4f}")
  print(f"sample: {neural.sample(seed=SAMPLE_SEED, max_chars=SAMPLE_CHARS)}")
  return 0


def _show_attention(model: GptModel) -> None:
  """For the last letter, the earlier letters each head of the last layer looks at most."""
  looks = model.looks(ATTENTION_TEXT)[-1]
  letters = ["⟨start⟩", *ATTENTION_TEXT][-looks.shape[-1] :]
  print(f"text: {ATTENTION_TEXT!r}; who the last letter {letters[-1]!r} looks at:")
  for head, row in enumerate(looks[:, -1]):
    top = torch.topk(row, SHOWN_LOOKS)
    shown = ", ".join(
      f"{letters[int(where)]!r} at {int(where)} ({float(share):.0%})"
      for share, where in zip(top.values, top.indices, strict=True)
    )
    print(f"  head {head}: {shown}")


def run_gpt() -> int:
  """Tiny GPT: steps chosen on validation, exam opened once at the end."""
  studied = _studied()
  if studied is None:
    return 1
  split, vocab = studied
  config = GptConfig(check_every=GPT_CHECK_EVERY)
  _, history = train_gpt(split_articles(split.train), vocab, config, steps=GPT_STEPS)
  best = _chosen_steps(history, len(history))
  model, _ = train_gpt(Split(split.train, ()), vocab, config, steps=best)
  print(f"parameters: {model.size():,}")
  _report(model, split)
  _show_attention(model)
  return 0


def main(arguments: list[str]) -> int:
  """Choose the experiment; the bigram is the default."""
  experiments = {
    "bigram": run_bigram,
    "network": run_network,
    "neural": run_neural,
    "bpe": run_bpe,
    "gpt": run_gpt,
  }
  name = arguments[0] if arguments else "bigram"
  if name not in experiments:
    print(f"lab: unknown experiment {name!r}; choose from {', '.join(experiments)}")
    return 1
  return experiments[name]()


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
