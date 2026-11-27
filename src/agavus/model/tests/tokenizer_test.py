"""Byte-level BPE on synthetic legal text: lossless, compact, turn tokens kept whole."""

from pathlib import Path

from agavus.model.tokenizer import (
  ASSISTANT,
  END,
  SPECIAL,
  USER,
  chars_per_token,
  decode,
  encode,
  load_tokenizer,
  save_tokenizer,
  train_tokenizer,
)

TEXTS = [
  "Art. 22. Constatada a prática de violência doméstica e familiar contra a mulher, "
  "o juiz poderá aplicar, de imediato, as seguintes medidas protetivas de urgência:",
  "Art. 19. As medidas protetivas de urgência poderão ser concedidas pelo juiz.",
  "§ 1º As medidas protetivas de urgência poderão ser concedidas de imediato.",
] * 20
UNSEEN = "Revogação? Não — só com decisão judicial! 🚫 Ñ ü ç 中文\n\ttabulação"
VOCAB = 400


def test_round_trip_is_exact_even_for_unseen_characters() -> None:
  tokenizer = train_tokenizer(TEXTS, VOCAB)
  for text in [*TEXTS[:3], UNSEEN, ""]:
    assert decode(tokenizer, encode(tokenizer, text)) == text


def test_merges_make_tokens_longer_than_one_character() -> None:
  tokenizer = train_tokenizer(TEXTS, VOCAB)
  assert chars_per_token(tokenizer, TEXTS[:3]) > len("ab")
  assert tokenizer.get_vocab_size() <= VOCAB


def test_turn_tokens_are_single_ids_and_come_first() -> None:
  tokenizer = train_tokenizer(TEXTS, VOCAB)
  assert [tokenizer.token_to_id(token) for token in SPECIAL] == [0, 1, 2]
  chat = f"{USER}O que diz o art. 22?{END}{ASSISTANT}Medidas protetivas.{END}"
  ids = encode(tokenizer, chat)
  assert ids.count(0) == len([END, END])
  assert decode(tokenizer, ids) == chat


def test_saved_tokenizer_encodes_the_same(tmp_path: Path) -> None:
  tokenizer = train_tokenizer(TEXTS, VOCAB)
  path = tmp_path / "tokenizer" / "bpe.json"
  save_tokenizer(tokenizer, path)
  assert encode(load_tokenizer(path), UNSEEN) == encode(tokenizer, UNSEEN)
  assert not list(path.parent.glob("*.partial"))
