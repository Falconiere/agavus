"""Bigram math: hand-worked values, normalization, determinism."""

import math

import numpy as np
import pytest

from agavus.language_model.bigram import UNKNOWN, build_vocab, train_bigram, uniform_loss

TEXT = "guarda guarda"
TEXT_VOCAB_SIZE = 8
MAX_CHARS = 15


def test_hand_worked_probability() -> None:
  # After "a": r, space, r, boundary. V = 6 characters + 2 reserved.
  model = train_bigram([TEXT])
  assert model.vocab.size == TEXT_VOCAB_SIZE
  assert model.probability("a", "r") == pytest.approx((2 + 1) / (4 + 8))


def test_hand_worked_loss() -> None:
  # V = 3; boundary->a (2/4), a->boundary (2/4).
  model = train_bigram(["a"])
  assert model.loss(["a"]) == pytest.approx(math.log(2))


def test_rows_sum_to_one() -> None:
  model = train_bigram([TEXT, "outra"])
  assert np.allclose(model.probs.sum(axis=1), 1.0)


def test_encode_decode_and_unknown() -> None:
  vocab = build_vocab([TEXT])
  assert vocab.decode(vocab.encode(TEXT)) == TEXT
  assert vocab.encode("z")[1] == UNKNOWN
  assert vocab.decode(vocab.encode("z")) == "\N{REPLACEMENT CHARACTER}"


def test_unseen_characters_have_finite_loss() -> None:
  assert math.isfinite(train_bigram([TEXT]).loss(["xyz"]))


def test_pairs_never_cross_texts() -> None:
  joined = train_bigram(["ab"]).probability("b", "a")
  separate = train_bigram(["ab", "ab"])
  # b is followed only by boundary, so b->a keeps smoothing mass alone.
  assert separate.probability("b", "a") == pytest.approx(1 / (2 + 4))
  assert joined == pytest.approx(1 / (1 + 4))


def test_learning_beats_uniform() -> None:
  model = train_bigram([TEXT] * 20)
  assert model.loss([TEXT]) < uniform_loss(model.vocab)


def test_sample_is_seeded_and_bounded() -> None:
  model = train_bigram([TEXT])
  first = model.sample(seed=3, max_chars=MAX_CHARS)
  assert first == model.sample(seed=3, max_chars=MAX_CHARS)
  assert len(first) <= MAX_CHARS


def test_invalid_training_inputs() -> None:
  with pytest.raises(ValueError, match="at least one"):
    train_bigram([])
  with pytest.raises(ValueError, match="smoothing"):
    train_bigram([TEXT], smoothing=0)
