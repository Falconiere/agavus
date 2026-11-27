"""Neural language model: examples, learning, determinism (synthetic text)."""

import torch

from agavus.language_model.bigram import BOUNDARY, build_vocab
from agavus.language_model.neural import NeuralConfig, examples

# Many threads only fight over work this small.
torch.set_num_threads(1)

TEXTS = ["guarda guarda", "a guarda do filho"]
CONFIG = NeuralConfig(context=3, vector=8, hidden=32, batch=32, rate=1e-2, check_every=50)
STEPS = 200


def test_examples_look_back_and_pad() -> None:
  vocab = build_vocab(["ab"])
  contexts, answers = examples(["ab"], vocab, CONFIG)
  a, b = vocab.encode("ab")[1:3]
  assert contexts.tolist() == [[BOUNDARY] * 3, [BOUNDARY, BOUNDARY, a], [BOUNDARY, a, b]]
  assert answers.tolist() == [a, b, BOUNDARY]


def test_examples_count_matches_the_bigram() -> None:
  _, answers = examples(TEXTS, build_vocab(TEXTS), CONFIG)
  assert len(answers) == sum(len(text) + 1 for text in TEXTS)
