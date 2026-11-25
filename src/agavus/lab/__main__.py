"""Run lab step 1: `uv run python -m agavus.lab`."""

import sys

from agavus.config.settings import load_settings
from agavus.lab.bigram import train_bigram, uniform_loss
from agavus.lab.data import load_articles, split_articles

SAMPLE_SEED = 7
SAMPLE_CHARS = 200


def main() -> int:
  """Train the bigram on statutes and report losses and a sample."""
  articles = load_articles(load_settings().data_dir)
  if not articles:
    print("lab: no statute articles — ingest statutes first")
    return 1
  split = split_articles(articles)
  model = train_bigram(split.train)
  print(f"articles: {len(split.train)} train / {len(split.holdout)} holdout")
  print(f"vocabulary: {model.vocab.size} tokens")
  print(f"loss uniform:  {uniform_loss(model.vocab):.4f}")
  print(f"loss train:    {model.loss(split.train):.4f}")
  print(f"loss holdout:  {model.loss(split.holdout):.4f}")
  print(f"sample: {model.sample(seed=SAMPLE_SEED, max_chars=SAMPLE_CHARS)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
