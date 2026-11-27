"""Run the lab: `uv run python -m agavus.language_model [experiment] [options]`.

Experiments: bigram (default), network, neural, bpe, agathe (see agathe_run.py),
fetch-wikipedia, fetch-federal, export.
"""

import sys
from collections.abc import Callable

import httpx
import torch

from agavus.config.settings import load_settings
from agavus.language_model.agathe_run import run_agathe
from agavus.language_model.bigram import (
  CharVocab,
  build_vocab,
  loss_per_character,
  train_bigram,
  uniform_loss,
)
from agavus.language_model.bpe import FIRST_MERGE, BpeVocab, train_bpe
from agavus.language_model.bundle import BUNDLE_FILE, local_texts, read_bundle, write_bundle
from agavus.language_model.data import Split, load_articles, split_articles
from agavus.language_model.extra import fetch_wikipedia, save_wikipedia
from agavus.language_model.federal import fetch_federal
from agavus.language_model.network import (
  CLUES,
  REFUSED,
  TEXTS,
  forward,
  start,
  train,
  train_single_neuron,
)
from agavus.language_model.neural import NeuralConfig
from agavus.language_model.neural import train as train_neural
from agavus.language_model.report import SAMPLE_CHARS, SAMPLE_SEED, THREADS, chosen_steps, report

HIDDEN = 8
NETWORK_SEED = 0
STEPS = 2000
NEURAL_STEPS = 4000
SHOWN_CHECKS = 8
# Chosen on validation articles (learning/14-word-pieces.md).
BIGRAM_MERGES = 1000
NEURAL_MERGES = 100
BPE_NEURAL_STEPS = 3000
SHOWN_PIECES = 20
EXAMPLE_WORDS = ("guarda", "guardar", "medida protetiva", "descumprimento")
# About as much text as the rest of the Civil Code, from many articles.
WIKIPEDIA_ARTICLES = 350
WIKIPEDIA_CHARS = 1500
WIKIPEDIA_TIMEOUT = 60.0
RATE = 0.5
SHOWN_STEPS = (0, 10, 50, 100, 200, 500, 1000, 2000)


def _statutes() -> Split | None:
  articles = load_articles(load_settings().data_dir)
  if not articles:
    print("lab: no statute articles — ingest statutes first")
    return None
  return split_articles(articles)


def run_bigram() -> int:
  """Train the bigram on statutes and report losses and a sample."""
  split = _statutes()
  if split is None:
    return 1
  model = train_bigram(split.train)
  print(f"articles: {len(split.train)} train / {len(split.holdout)} holdout")
  print(f"vocabulary: {model.vocab.size} tokens")
  print(f"loss uniform:  {uniform_loss(model.vocab):.4f}")
  report(model, split)
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
  best = chosen_steps(history, SHOWN_CHECKS)
  model, _ = train_neural(Split(split.train, ()), vocab, config, steps=best)
  print(f"loss bigram holdout: {train_bigram(split.train).loss(split.holdout):.4f}")
  report(model, split)
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


def run_fetch_wikipedia() -> int:
  """Download the openings of Wikipedia articles into data/lab/ (CC BY-SA 4.0)."""
  data_dir = load_settings().data_dir
  with httpx.Client(timeout=WIKIPEDIA_TIMEOUT) as client:
    found = fetch_wikipedia(client, articles=WIKIPEDIA_ARTICLES, chars=WIKIPEDIA_CHARS)
  path = save_wikipedia(data_dir, found)
  print(f"saved {len(found)} articles, {sum(len(a.text) for a in found):,} characters: {path}")
  return 0


def run_fetch_federal() -> int:
  """Cache public-domain Planalto articles, using the exported statutes as the leak guard."""
  data_dir = load_settings().data_dir
  try:
    statutes = read_bundle(data_dir / BUNDLE_FILE).statutes
    with httpx.Client(timeout=60) as client:
      fetch_federal(data_dir, client, statutes)
  except (OSError, ValueError, httpx.HTTPError) as error:
    print(f"lab: federal fetch failed: {error}")
    return 1
  return 0


def run_export() -> int:
  """Write the allowlisted lab text to one file, to inspect and then upload."""
  data_dir = load_settings().data_dir
  texts = local_texts(data_dir)
  sha = write_bundle(data_dir / BUNDLE_FILE, texts)
  print(f"statutes: {len(texts.statutes)} articles")
  for name, found in texts.extras.items():
    print(f"{name}: {len(found)} texts, {sum(map(len, found)):,} characters")
  print(f"saved: {data_dir / BUNDLE_FILE}\nsha256: {sha}")
  return 0


def main(arguments: list[str]) -> int:
  """Choose the experiment; the bigram is the default."""
  experiments: dict[str, Callable[[list[str]], int]] = {
    "bigram": lambda _: run_bigram(),
    "network": lambda _: run_network(),
    "neural": lambda _: run_neural(),
    "bpe": lambda _: run_bpe(),
    "agathe": run_agathe,
    "fetch-federal": lambda _: run_fetch_federal(),
    "fetch-wikipedia": lambda _: run_fetch_wikipedia(),
    "export": lambda _: run_export(),
  }
  name = arguments[0] if arguments else "bigram"
  if name not in experiments:
    print(f"lab: unknown experiment {name!r}; choose from {', '.join(experiments)}")
    return 1
  return experiments[name](arguments[1:])


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
