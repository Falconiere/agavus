"""Small real CPU encoders and tokenizers for integration checks."""

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel

from agavus.classify.learned.model import LearnedModel


def tiny_tokenizer() -> Tokenizer:
  """Tokenizer with all boundary and fixture tokens, requiring no network."""
  vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "pedido": 4, "negado": 5, "sim": 6}
  tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
  tokenizer.pre_tokenizer = Whitespace()
  return tokenizer


def tiny_model() -> LearnedModel:
  """One-layer CPU Bert with deterministic initial weights."""
  torch.manual_seed(7)
  config = BertConfig(
    vocab_size=7,
    hidden_size=8,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=16,
    max_position_embeddings=512,
    pad_token_id=0,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
  )
  return LearnedModel(BertModel(config), config.hidden_size, ("F", "M", "O"))
