"""BPE: hand-worked merges, lossless round trips, fair comparison."""

import pytest

from agavus.language_model.bigram import CharVocab, build_vocab, loss_per_character, train_bigram
from agavus.language_model.bpe import FIRST_MERGE, BpeVocab, byte_ids, glue, train_bpe

TEXT = "guarda guarda"


def test_glue_goes_left_to_right() -> None:
  assert glue([1, 1, 1, 2], (1, 1), 9) == [9, 1, 2]
  assert glue([], (1, 1), 9) == []


def test_first_merge_is_the_most_frequent_pair() -> None:
  vocab = train_bpe(["ababab ac"], merges=1)
  assert vocab.piece(FIRST_MERGE) == "ab"


def test_frequent_word_becomes_one_piece() -> None:
  vocab = train_bpe([TEXT], merges=50)
  assert "guarda" in {vocab.piece(token) for token in vocab.encode(TEXT)}
  assert len(vocab.encode(TEXT)) < len(byte_ids(TEXT))


@pytest.mark.parametrize(
  "text", ["", " ", TEXT, "ação, § 2º — «x»", "日本語 🙂", "a\n\tb  c \r\n", "nunca visto"]
)
def test_round_trip_is_lossless(text: str) -> None:
  for vocab in (train_bpe([TEXT, "a ação do juiz, a ação"], merges=30), BpeVocab(())):
    assert vocab.decode(vocab.encode(text)) == text


def test_training_is_deterministic_and_stops() -> None:
  assert train_bpe([TEXT], merges=20) == train_bpe([TEXT], merges=20)
  assert train_bpe(["a"], merges=20).merges == ()


def test_loss_per_character_leaves_character_models_unchanged() -> None:
  vocab: CharVocab = build_vocab([TEXT])
  loss = train_bigram([TEXT]).loss([TEXT])
  assert loss_per_character(loss, [TEXT], vocab) == pytest.approx(loss)
