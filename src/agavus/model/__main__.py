"""Run the lab: `uv run python -m agavus.model [experiment] [options]`.

Experiments: bigram (default), network, neural, bpe, agathe (see agathe_run.py),
chat, recall, train-tokenizer, fetch-wikipedia, fetch-federal, fetch-planalto, export,
build-sft, build-search, build-sft-context, sft.
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import httpx
import torch

from agavus.config.settings import load_settings
from agavus.model.agathe import placed
from agavus.model.agathe_run import run_agathe, save_run
from agavus.model.bigram import (
  CharVocab,
  build_vocab,
  loss_per_character,
  train_bigram,
  uniform_loss,
)
from agavus.model.bpe import FIRST_MERGE, BpeVocab, train_bpe
from agavus.model.bundle import BUNDLE_FILE, local_texts, read_bundle, with_local, write_bundle
from agavus.model.checkpoint import load_checkpoint
from agavus.model.data import Split, load_articles, load_case_texts, split_articles
from agavus.model.extra import fetch_wikipedia, save_wikipedia, without_leaks
from agavus.model.federal import fetch_federal
from agavus.model.law_search import SEARCH_FILE, build_index, index_counts
from agavus.model.network import (
  CLUES,
  REFUSED,
  TEXTS,
  forward,
  start,
  train,
  train_single_neuron,
)
from agavus.model.neural import NeuralConfig
from agavus.model.neural import train as train_neural
from agavus.model.planalto import (
  PLANALTO_DIR,
  Crawl,
  discover,
  fetch_planalto,
  reextract_planalto,
)
from agavus.model.recall import recall, sample_texts
from agavus.model.report import SAMPLE_CHARS, SAMPLE_SEED, THREADS, chosen_steps, report
from agavus.model.sft import (
  SFT_DIR,
  Conversation,
  TuneConfig,
  load_conversations,
  masked_loss,
  save_conversations,
  tune,
)
from agavus.model.sft_context import NOT_FOUND, contextualize, teacher_for_training
from agavus.model.sft_rules import law_of, load_rule_sources, rule_conversations, split_by_law
from agavus.model.teacher import run_teach
from agavus.model.tokenizer import (
  TOKENIZER_FILE,
  chars_per_token,
  save_tokenizer,
  train_tokenizer,
)
from agavus.server.__main__ import serve

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
SFT_STEPS = 2000
# What load_checkpoint needs from the base run to load the tuned one.
BASE_FIELDS = ("config", "data_sha256", "tokenizer_sha256", "extras")


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


def run_fetch_planalto(arguments: list[str]) -> int:
  """Refresh every indexed federal law; `--offline` re-extracts the saved captures instead."""
  data_dir = load_settings().data_dir
  try:
    statutes = read_bundle(data_dir / BUNDLE_FILE).statutes
    if arguments == ["--offline"]:
      counts = reextract_planalto(data_dir, statutes)
      print("planalto offline: " + ", ".join(f"{key} {value}" for key, value in counts.items()))
      return 0
    captured = sorted((data_dir / PLANALTO_DIR / "indexes").glob("*.html"))
    with httpx.Client(timeout=60) as client:
      crawl = Crawl(client)
      laws = discover(crawl, captured)
      print(f"planalto: {len(laws)} law pages from {len(captured)} saved listings", flush=True)
      counts = fetch_planalto(data_dir, crawl, statutes, laws)
  except (OSError, ValueError, httpx.HTTPError) as error:
    print(f"lab: planalto fetch failed: {error}")
    return 1
  print("planalto: " + ", ".join(f"{key} {value}" for key, value in counts.items()))
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


def run_train_tokenizer(names: list[str]) -> int:
  """Train the BPE tokenizer on statute training articles plus the named texts, leak-free."""
  data_dir = load_settings().data_dir
  try:
    texts = with_local(data_dir, names or ["civil", "federal", "planalto", "djen"])
  except (OSError, ValueError) as error:
    print(f"lab: tokenizer failed: {error}")
    return 1
  split = split_articles(texts.statutes)
  sources = {name: without_leaks(found, texts.statutes) for name, found in texts.extras.items()}
  corpus = [*split.train, *(text for found in sources.values() for text in found)]
  tokenizer = train_tokenizer(corpus)
  save_tokenizer(tokenizer, data_dir / TOKENIZER_FILE)
  print(f"tokenizer: {tokenizer.get_vocab_size():,} tokens from {sum(map(len, corpus)):,} chars")
  for name, found in [("exam", split.holdout), *sources.items()]:
    print(f"  {name}: {chars_per_token(tokenizer, sample_texts(found, 2000)):.2f} chars/token")
  print(f"saved: {data_dir / TOKENIZER_FILE}")
  return 0


def run_recall(arguments: list[str]) -> int:
  """Probe how much of a text set a checkpoint reproduces verbatim."""
  data_dir = load_settings().data_dir
  parser = argparse.ArgumentParser(prog="python -m agavus.model recall")
  parser.add_argument("--run", type=Path, default=data_dir / "lab/runs/long-civil")
  parser.add_argument("--set", choices=("train", "exam", "djen"), default="exam")
  parser.add_argument("--limit", type=int, default=200)
  parser.add_argument("--prompt", type=int, default=64, help="prompt characters")
  parser.add_argument("--answer", type=int, default=64, help="answer characters")
  parser.add_argument("--tokenizer", type=Path, default=data_dir / TOKENIZER_FILE)
  options = parser.parse_args(arguments)
  torch.set_num_threads(THREADS)
  try:
    model = load_checkpoint(options.run, data_dir / BUNDLE_FILE, options.tokenizer, data_dir)
    split = split_articles(read_bundle(data_dir / BUNDLE_FILE).statutes)
  except (OSError, ValueError) as error:
    print(f"lab: recall failed: {error}")
    return 1
  if options.set == "djen":
    texts = load_case_texts(data_dir)
  else:
    texts = split.train if options.set == "train" else split.holdout
  found = recall(
    model,
    sample_texts(texts, options.limit),
    prompt_chars=options.prompt,
    answer_chars=options.answer,
  )
  print(
    f"recall {options.set}: texts {found.texts}, exact {found.exact:.1%}, "
    f"matched {found.matched:.1%}, loss {found.loss:.4f}"
  )
  return 0


def _kinds(conversations: list[Conversation]) -> str:
  """Conversation counts per source, as 'source count, ...'."""
  counts = sorted(Counter(conversation.source for conversation in conversations).items())
  return ", ".join(f"{kind} {count}" for kind, count in counts)


def run_build_sft() -> int:
  """Rule conversations from the Planalto crawl, whole laws held out, saved for `sft`."""
  data_dir = load_settings().data_dir
  try:
    cards, articles = load_rule_sources(data_dir)
  except (OSError, KeyError, ValueError) as error:
    print(f"lab: build-sft failed: {error}")
    return 1
  conversations = rule_conversations(cards, articles, seed=0)
  if not conversations:
    print("lab: no Planalto laws — run `python -m agavus.model fetch-planalto` first")
    return 1
  print(f"build-sft: {len(cards)} law cards, {sum(map(len, articles.values()))} articles")
  for name, part in zip(("rules-train", "rules-eval"), split_by_law(conversations), strict=True):
    save_conversations(data_dir / SFT_DIR / f"{name}.jsonl", part)
    laws = len({law_of(conversation) for conversation in part})
    print(f"{name}: {len(part)} conversations from {laws} laws ({_kinds(part)})")
  return 0


def run_build_search() -> int:
  """Index every crawled Planalto article for BM25 search."""
  data_dir = load_settings().data_dir
  try:
    articles, laws = index_counts(build_index(data_dir))
  except (OSError, KeyError, ValueError, sqlite3.Error) as error:
    print(f"lab: build-search failed: {error}")
    return 1
  print(f"build-search: {articles} articles from {laws} laws: {data_dir / SEARCH_FILE}")
  return 0


def run_build_sft_context(arguments: list[str]) -> int:
  """Rule (and teacher) conversations with their search hits in context, for `sft`."""
  data_dir = load_settings().data_dir
  parser = argparse.ArgumentParser(prog="python -m agavus.model build-sft-context")
  parser.add_argument("--teacher", type=Path, help="teacher JSONL, joins the train side only")
  parser.add_argument("--rules", type=int, help="seeded sample of training rule conversations")
  parser.add_argument("--eval", type=int, help="seeded sample of held-out rule conversations")
  options = parser.parse_args(arguments)
  db, folder = data_dir / SEARCH_FILE, data_dir / SFT_DIR
  try:
    train = load_conversations(folder / "rules-train.jsonl")
    held_out = load_conversations(folder / "rules-eval.jsonl")
    # Search costs ~50 ms per conversation; a seeded sample keeps the build under an hour.
    train = list(sample_texts(train, options.rules or len(train)))
    evaluated = list(sample_texts(held_out, options.eval or len(held_out)))
    if options.teacher and not options.teacher.is_file():
      print(f"build-sft-context: no teacher file at {options.teacher}, skipped")
    elif options.teacher:
      train += teacher_for_training(load_conversations(options.teacher), held_out)
    if not db.is_file():
      raise ValueError(f"no search index at {db}")
    for name, part in (("context-train", train), ("context-eval", evaluated)):
      found = contextualize(part, db, seed=0, negatives=0.1)
      save_conversations(folder / f"{name}.jsonl", found)
      negatives = sum(conversation.turns[-1][1] == NOT_FOUND for conversation in found)
      print(f"{name}: {len(found)} of {len(part)}, {negatives} not found ({_kinds(found)})")
  except (OSError, ValueError, sqlite3.Error) as error:
    print(f"lab: build-sft-context failed: {error} — run build-sft and build-search first?")
    return 1
  return 0


def run_sft(arguments: list[str]) -> int:
  """Chat-tune a saved Agathe run on the rule conversations; the result serves as a chat."""
  data_dir = load_settings().data_dir
  defaults = TuneConfig()
  parser = argparse.ArgumentParser(prog="python -m agavus.model sft")
  parser.add_argument("--run", type=Path, required=True, help="base run folder")
  parser.add_argument("--out", type=Path, required=True, help="folder for the tuned run")
  parser.add_argument("--steps", type=int, default=SFT_STEPS)
  parser.add_argument("--rate", type=float, default=defaults.rate, help="AdamW learning rate")
  parser.add_argument("--batch", type=int, default=defaults.batch, help="conversations per step")
  parser.add_argument("--device", default="cpu", help="cpu, cuda, ...")
  parser.add_argument("--bf16", action="store_true", help="bfloat16 mixed precision")
  parser.add_argument("--tokenizer", type=Path, default=data_dir / TOKENIZER_FILE)
  parser.add_argument("--train", type=Path, default=data_dir / SFT_DIR / "rules-train.jsonl")
  parser.add_argument("--eval", type=Path, default=data_dir / SFT_DIR / "rules-eval.jsonl")
  options = parser.parse_args(arguments)
  if options.steps < 1 or options.batch < 1 or not options.rate > 0:
    parser.error("--steps, --batch and --rate must be positive")
  torch.set_num_threads(THREADS)
  try:
    base = load_checkpoint(options.run, data_dir / BUNDLE_FILE, options.tokenizer, data_dir)
    record = json.loads((options.run / "run.json").read_bytes())
    train_set = load_conversations(options.train)
    held_out = load_conversations(options.eval)
  except (OSError, ValueError) as error:
    print(f"lab: sft failed: {error} — run `python -m agavus.model build-sft` first?")
    return 1
  model = placed(base, replace(base.config, device=options.device, bf16=options.bf16))
  settings = replace(defaults, rate=options.rate, batch=options.batch)
  model, history = tune(model, train_set, settings, steps=options.steps, held_out=held_out)
  loss = masked_loss(model, model.tokenizer, held_out)
  print(f"masked loss held-out: {loss:.4f} nats per answer token")
  tuned: dict[str, object] = {key: record[key] for key in BASE_FIELDS if key in record}
  tuned |= {
    "chat": True,
    "base": options.run.resolve().name,
    "parameters": model.size(),
    "steps": options.steps,
    "tune": {"rate": settings.rate, "batch": settings.batch, "seed": settings.seed},
    "conversations": {"train": len(train_set), "held_out": len(held_out)},
    "history": history,
    "masked_loss_held_out": loss,
  }
  save_run(options.out, model, tuned)
  return 0


def main(arguments: list[str]) -> int:
  """Choose the experiment; the bigram is the default."""
  experiments: dict[str, Callable[[list[str]], int]] = {
    "bigram": lambda _: run_bigram(),
    "network": lambda _: run_network(),
    "neural": lambda _: run_neural(),
    "bpe": lambda _: run_bpe(),
    "agathe": run_agathe,
    "chat": serve,
    "recall": run_recall,
    "train-tokenizer": run_train_tokenizer,
    "fetch-federal": lambda _: run_fetch_federal(),
    "fetch-planalto": run_fetch_planalto,
    "fetch-wikipedia": lambda _: run_fetch_wikipedia(),
    "export": lambda _: run_export(),
    "build-sft": lambda _: run_build_sft(),
    "build-search": lambda _: run_build_search(),
    "build-sft-context": run_build_sft_context,
    "sft": run_sft,
    "teach": run_teach,
  }
  name = arguments[0] if arguments else "bigram"
  if name not in experiments:
    print(f"lab: unknown experiment {name!r}; choose from {', '.join(experiments)}")
    return 1
  return experiments[name](arguments[1:])


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
