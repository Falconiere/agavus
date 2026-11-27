"""Restore a saved Agathe run for CPU inference, checking its text bundle and tokenizer."""

import json
import pickle
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import torch

from agavus.model.agathe import AgatheConfig, AgatheModel, check_config, start
from agavus.model.bundle import LOCAL, Texts, bundle_bytes, digest, merged, read_bundle
from agavus.model.tokenizer import load_tokenizer

DIMENSIONS = ("context", "width", "heads", "kv_heads", "layers")


def _inputs_sha256(bundle: Path, texts: Texts, extras: Sequence[str], data_dir: Path | None) -> str:
  """Runs with local texts hash the bundle plus the local texts they trained on."""
  if data_dir is None or not set(extras) & LOCAL.keys():
    return digest(bundle)
  return sha256(bundle_bytes(merged(texts, data_dir, extras))).hexdigest()


def _hash(record: dict[str, object], key: str) -> str:
  """A recorded SHA-256, refused unless it is one."""
  found = record[key]
  if not isinstance(found, str) or len(bytes.fromhex(found)) != sha256().digest_size:
    raise ValueError(f"SHA-256 inválido em {key}")
  return found


def load_checkpoint(
  run: Path, bundle: Path, tokenizer: Path, data_dir: Path | None = None
) -> AgatheModel:
  """Validate the four artifacts before exposing the model to a chat session."""
  record_path = run / "run.json"
  try:
    record = json.loads(record_path.read_bytes())
    config = record["config"]
    dimensions = {key: config[key] for key in DIMENSIONS}
    if any(type(value) is not int for value in dimensions.values()):
      raise ValueError("dimensões inválidas")
    if type(config["rope_base"]) is not float or not config["rope_base"] > 0:
      raise ValueError("base RoPE inválida")
    settings = AgatheConfig(**dimensions, rope_base=config["rope_base"])
    check_config(settings)
    expected, words = _hash(record, "data_sha256"), _hash(record, "tokenizer_sha256")
  except (OSError, TypeError, KeyError, ValueError) as error:
    raise ValueError(f"Confira {record_path}: {error}") from error

  try:
    texts = read_bundle(bundle)
    if _inputs_sha256(bundle, texts, record.get("extras", ()), data_dir) != expected:
      raise ValueError(f"o SHA-256 difere do registrado em {record_path}")
  except (OSError, AttributeError, KeyError, TypeError, ValueError) as error:
    raise ValueError(f"Confira {bundle}: {error}") from error

  try:
    if digest(tokenizer) != words:
      raise ValueError(f"o SHA-256 difere do registrado em {record_path}")
    model = start(load_tokenizer(tokenizer), settings)
  except (OSError, ValueError) as error:
    raise ValueError(f"Confira {tokenizer}: {error}") from error
  weights = run / "model.pt"
  try:
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(saved, list):
      raise ValueError("lista de pesos esperada")
    with torch.no_grad():
      for parameter, value in zip(model.parameters(), saved, strict=True):
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
          raise ValueError("dimensões incompatíveis com os parâmetros do modelo")
        if value.dtype != parameter.dtype or not torch.isfinite(value).all():
          raise ValueError("pesos inválidos")
        parameter.copy_(value)
  except (
    OSError,
    RuntimeError,
    ValueError,
    LookupError,
    EOFError,
    pickle.UnpicklingError,
  ) as error:
    raise ValueError(f"Confira {weights}: {error}") from error
  return model
