"""Safe local model artifacts, separate from retrieval and live classifiers."""

import math
import sqlite3
from pathlib import Path
from typing import cast

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from transformers import AutoModel, PreTrainedModel

from agavus.classify.learned.model import LearnedModel
from agavus.retrieve.encoder import REPO, REVISION
from agavus.taxonomy.areas import AREAS


def prepare_pretrained(data_dir: Path) -> Path:
  """Download only pinned trainable assets, never touch the ONNX encoder directory."""
  folder = data_dir / "experiments" / "learned" / "pretrained" / REVISION
  for filename in ("config.json", "model.safetensors", "tokenizer.json"):
    hf_hub_download(REPO, filename, revision=REVISION, local_dir=folder)
  return folder


def initialize(folder: Path, seed: int) -> tuple[LearnedModel, Tokenizer]:
  """Each ablation starts from the same pretrained weights and seeded heads."""
  torch.manual_seed(seed)
  encoder = AutoModel.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
  model = LearnedModel(encoder, encoder.config.hidden_size, tuple(sorted(AREAS)))
  tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
  tokenizer.no_truncation()
  tokenizer.no_padding()
  return model, tokenizer


def save_checkpoint(
  model: LearnedModel,
  folder: Path,
  tokenizer: Tokenizer,
  temperatures: tuple[float, float] = (1.0, 1.0),
) -> None:
  """Save tokenizer, encoder, heads and calibrated task temperatures without pickle."""
  folder.mkdir(parents=True, exist_ok=True)
  encoder = cast(PreTrainedModel, model.encoder)
  encoder.save_pretrained(str(folder / "encoder"), safe_serialization=True)
  tokenizer.save(str(folder / "tokenizer.json"))
  heads = {
    key: value.detach().cpu().contiguous()
    for key, value in model.state_dict().items()
    if not key.startswith("encoder.")
  }
  save_file(heads, str(folder / "heads.safetensors"))
  with sqlite3.connect(folder / "checkpoint.db") as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS classes (position INTEGER PRIMARY KEY, label TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS temperatures (task TEXT PRIMARY KEY, value REAL)")
    conn.execute("DELETE FROM classes")
    conn.executemany("INSERT INTO classes VALUES (?,?)", enumerate(model.classes))
    conn.executemany(
      "INSERT OR REPLACE INTO temperatures VALUES (?,?)",
      zip(("area", "relation"), temperatures, strict=True),
    )


def load_checkpoint(folder: Path) -> tuple[LearnedModel, Tokenizer, tuple[float, float]]:
  """CPU is always a valid reload and inference target."""
  encoder = AutoModel.from_pretrained(str(folder / "encoder"), local_files_only=True)
  with sqlite3.connect(folder / "checkpoint.db") as conn:
    classes = tuple(
      str(row[0]) for row in conn.execute("SELECT label FROM classes ORDER BY position")
    )
    temperatures = {
      str(k): float(v) for k, v in conn.execute("SELECT task,value FROM temperatures")
    }
  model = LearnedModel(encoder, encoder.config.hidden_size, classes)
  heads = load_file(str(folder / "heads.safetensors"))
  expected = {key for key in model.state_dict() if not key.startswith("encoder.")}
  if set(heads) != expected or set(classes) != set(AREAS) or len(set(classes)) != len(classes):
    raise ValueError("invalid checkpoint classes or head tensors")
  if any(not math.isfinite(t) or t <= 0 for t in temperatures.values()):
    raise ValueError("invalid calibrated temperatures")
  model.load_state_dict(heads, strict=False)
  model.eval()
  tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
  tokenizer.no_truncation()
  tokenizer.no_padding()
  return (
    model,
    tokenizer,
    (temperatures["area"], temperatures["relation"]),
  )
