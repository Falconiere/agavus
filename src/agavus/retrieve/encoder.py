"""Multilingual sentence encoder on CPU via ONNX (weights are optional).

Off-the-shelf weights, pinned by revision; no training. Its vectors support question routing
and statute ranking. The quoted text comes from the corpus.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from tokenizers import Tokenizer

FloatArray = NDArray[np.float64]
REPO = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_VERSION = f"multilingual-e5-small@{REVISION[:8]}"
MODEL_FILE = "onnx/model.onnx"
TOKENIZER_FILE = "onnx/tokenizer.json"
MAX_TOKENS = 512
BATCH = 16


class Encoder(Protocol):
  """Anything that maps texts to unit vectors."""

  def encode_queries(self, texts: Sequence[str]) -> FloatArray:
    """One unit vector per question."""
    ...

  def encode_passages(self, texts: Sequence[str]) -> FloatArray:
    """One unit vector per passage."""
    ...


def encoder_dir(data_dir: Path) -> Path:
  """Where the pinned weights live (gitignored, rebuildable)."""
  return data_dir / "models" / "encoder" / REVISION


def download_encoder(data_dir: Path) -> Path:
  """Fetch the pinned weights from Hugging Face; idempotent."""
  folder = encoder_dir(data_dir)
  for name in (MODEL_FILE, TOKENIZER_FILE):
    hf_hub_download(REPO, name, revision=REVISION, local_dir=folder)
  return folder


def onnx_model(model: Path, tokenizer: Path) -> tuple[Tokenizer, InferenceSession]:
  """An XLM-R tokenizer (truncated to MAX_TOKENS, padded with <pad>) and its CPU session."""
  loaded = Tokenizer.from_file(str(tokenizer))
  loaded.enable_truncation(MAX_TOKENS)
  loaded.enable_padding(pad_id=1, pad_token="<pad>")
  return loaded, InferenceSession(str(model), providers=["CPUExecutionProvider"])


class OnnxEncoder:
  """E5 encoder: prefix, mean pooling over tokens, L2 normalization."""

  def __init__(self, folder: Path) -> None:
    self._tokenizer, self._session = onnx_model(folder / MODEL_FILE, folder / TOKENIZER_FILE)

  def encode_queries(self, texts: Sequence[str]) -> FloatArray:
    """One unit vector per question."""
    return self._encode([f"query: {text}" for text in texts])

  def encode_passages(self, texts: Sequence[str]) -> FloatArray:
    """One unit vector per passage."""
    return self._encode([f"passage: {text}" for text in texts])

  def _encode(self, texts: Sequence[str]) -> FloatArray:
    return np.concatenate(
      [self._batch(texts[start : start + BATCH]) for start in range(0, len(texts), BATCH)]
    )

  def _batch(self, texts: Sequence[str]) -> FloatArray:
    encoded = self._tokenizer.encode_batch(list(texts))
    ids = np.array([item.ids for item in encoded], dtype=np.int64)
    mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
    hidden = np.asarray(self._session.run(["last_hidden_state"], feed)[0], dtype=np.float64)
    pooled = (hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[:, None]
    return np.asarray(pooled / np.linalg.norm(pooled, axis=1, keepdims=True), dtype=np.float64)


def load_encoder(data_dir: Path) -> Encoder | None:
  """The local encoder; None when the weights are not installed."""
  folder = encoder_dir(data_dir)
  if not (folder / MODEL_FILE).exists() or not (folder / TOKENIZER_FILE).exists():
    return None
  return OnnxEncoder(folder)
