"""Agathe's byte-level BPE tokenizer (docs/llm-lab.md §9.1).

The `tokenizers` library learns the merges; the model stays ours. Byte-level pieces make
every text encodable and decoding exact, so no Unicode normalization is applied. Special
tokens mark conversation turns; `<|end|>` also separates documents during pretraining.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

END, USER, ASSISTANT = "<|end|>", "<|user|>", "<|assistant|>"
SPECIAL = (END, USER, ASSISTANT)
VOCAB_SIZE = 32_768
TOKENIZER_FILE = Path("lab") / "tokenizer" / "agathe-bpe.json"


def train_tokenizer(texts: Iterable[str], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
  """Learn byte-level BPE merges from `texts`; the special tokens get the first ids."""
  tokenizer = Tokenizer(models.BPE())
  tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
  tokenizer.decoder = decoders.ByteLevel()
  trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=list(SPECIAL),
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    min_frequency=2,
    show_progress=False,
  )
  tokenizer.train_from_iterator(texts, trainer=trainer)
  return tokenizer


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
  """Token ids for `text`, without adding any special token."""
  return list(tokenizer.encode(text, add_special_tokens=False).ids)


def decode(tokenizer: Tokenizer, ids: Sequence[int]) -> str:
  """Text for token ids, special tokens included as written."""
  return str(tokenizer.decode(list(ids), skip_special_tokens=False))


def chars_per_token(tokenizer: Tokenizer, texts: Sequence[str]) -> float:
  """How many characters one token carries on average over `texts`."""
  tokens = sum(len(encode(tokenizer, text)) for text in texts)
  return sum(map(len, texts)) / max(1, tokens)


def save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
  """Write the tokenizer as one JSON file, whole or not at all."""
  path.parent.mkdir(parents=True, exist_ok=True)
  staged = path.with_suffix(".partial")
  tokenizer.save(str(staged))
  staged.replace(path)


def load_tokenizer(path: Path) -> Tokenizer:
  """Read a tokenizer saved by `save_tokenizer`."""
  return Tokenizer.from_file(str(path))
