"""Pretrained multilingual reranker on CPU via ONNX: how well a passage answers a question.

Off-the-shelf weights pinned by revision, no training: BAAI/bge-reranker-v2-m3 (Apache-2.0),
as the onnx-community int8 export. A cross-encoder reads the question and the passage
together and returns one relevance logit; the chat quotes the best passage verbatim.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from huggingface_hub import hf_hub_download

from agavus.retrieve.encoder import onnx_model

REPO = "onnx-community/bge-reranker-v2-m3-ONNX"
REVISION = "6f5ff65298512715a1e669753bc754d2bc8f367b"
MODEL_VERSION = f"bge-reranker-v2-m3-int8@{REVISION[:8]}"
MODEL_FILE = "onnx/model_int8.onnx"
TOKENIZER_FILE = "tokenizer.json"
BATCH = 16


class Reranker(Protocol):
  """Anything that scores passages against a question; higher is more relevant."""

  def scores(self, question: str, passages: Sequence[str]) -> list[float]:
    """One relevance logit per passage."""
    ...


def reranker_dir(data_dir: Path) -> Path:
  """Where the pinned weights live (gitignored, rebuildable)."""
  return data_dir / "models" / "reranker" / REVISION


def download_reranker(data_dir: Path) -> Path:
  """Fetch the pinned weights from Hugging Face; idempotent."""
  folder = reranker_dir(data_dir)
  for name in (MODEL_FILE, TOKENIZER_FILE):
    hf_hub_download(REPO, name, revision=REVISION, local_dir=folder)
  return folder


class OnnxReranker:
  """The cross-encoder: (question, passage) pairs in, one logit each out."""

  def __init__(self, model: Path, tokenizer: Path) -> None:
    # The same XLM-R tokenizer setup as the product's E5 encoder: truncated to 512 tokens.
    self._tokenizer, self._session = onnx_model(model, tokenizer)

  def scores(self, question: str, passages: Sequence[str]) -> list[float]:
    """One relevance logit per passage, in order."""
    found: list[float] = []
    for start in range(0, len(passages), BATCH):
      pairs = [(question, passage) for passage in passages[start : start + BATCH]]
      encoded = self._tokenizer.encode_batch(pairs)
      feed = {
        "input_ids": np.array([item.ids for item in encoded], dtype=np.int64),
        "attention_mask": np.array([item.attention_mask for item in encoded], dtype=np.int64),
      }
      logits = np.asarray(self._session.run(None, feed)[0], dtype=np.float64)
      found.extend(float(value) for value in logits.reshape(-1))
    return found


def load_reranker(data_dir: Path) -> Reranker | None:
  """The local reranker; None when the weights are not installed."""
  folder = reranker_dir(data_dir)
  if not (folder / MODEL_FILE).exists() or not (folder / TOKENIZER_FILE).exists():
    return None
  return OnnxReranker(folder / MODEL_FILE, folder / TOKENIZER_FILE)
