"""Restore a saved Agathe run for CPU inference using its original text bundle."""

import json
import pickle
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import torch

from agavus.model.agathe import AgatheConfig, AgatheModel, start
from agavus.model.bigram import build_vocab
from agavus.model.bundle import LOCAL, Texts, bundle_bytes, digest, merged, read_bundle
from agavus.model.data import split_articles


def _inputs_sha256(bundle: Path, texts: Texts, extras: Sequence[str], data_dir: Path | None) -> str:
  """Runs with local texts hash the bundle plus the local texts they trained on."""
  if data_dir is None or not set(extras) & LOCAL.keys():
    return digest(bundle)
  return sha256(bundle_bytes(merged(texts, data_dir, extras))).hexdigest()


def load_checkpoint(run: Path, bundle: Path, data_dir: Path | None = None) -> AgatheModel:
  """Validate the three artifacts before exposing the model to a chat session."""
  record_path = run / "run.json"
  try:
    record = json.loads(record_path.read_bytes())
    config = record["config"]
    dimensions = {key: config[key] for key in ("context", "width", "heads", "layers")}
    if any(type(value) is not int or value < 1 for value in dimensions.values()):
      raise ValueError("dimensões inválidas")
    if dimensions["width"] % dimensions["heads"]:
      raise ValueError("largura incompatível com o número de cabeças")
    expected = record["data_sha256"]
    if not isinstance(expected, str) or len(bytes.fromhex(expected)) != sha256().digest_size:
      raise ValueError("SHA-256 inválido")
  except (OSError, TypeError, KeyError, ValueError) as error:
    raise ValueError(f"Confira {record_path}: {error}") from error

  try:
    texts = read_bundle(bundle)
    if _inputs_sha256(bundle, texts, record.get("extras", ()), data_dir) != expected:
      raise ValueError(f"o SHA-256 difere do registrado em {record_path}")
    training = split_articles(texts.statutes).train
    if not training:
      raise ValueError("faltam artigos de treino")
  except (OSError, AttributeError, KeyError, TypeError, ValueError) as error:
    raise ValueError(f"Confira {bundle}: {error}") from error

  model = start(build_vocab(training), AgatheConfig(**dimensions))
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
