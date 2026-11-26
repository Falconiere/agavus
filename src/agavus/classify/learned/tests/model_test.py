"""CPU integration tests for learned encoder training and persistence."""

import math
import sqlite3
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from agavus.classify.learned.checkpoint import load_checkpoint, save_checkpoint
from agavus.classify.learned.model import tensors
from agavus.classify.learned.train import TokenizedSample, build_optimizer, train_step
from agavus.classify.learned.types import Sample

from .fixtures import tiny_model, tiny_tokenizer

WINDOWS = ((2, 4, 3), (2, 5, 3))


def test_masked_padding_and_document_window_mean_are_invariant() -> None:
  model = tiny_model()
  model.eval()
  ids, mask = tensors([WINDOWS[0]], "cpu")
  padded_ids = torch.tensor([[2, 4, 3, 0, 0]], dtype=torch.long)
  padded_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
  with torch.no_grad():
    single = model.logits(ids, mask, "area")
    padded = model.logits(padded_ids, padded_mask, "area")
    rows = [model.logits(*tensors([window], "cpu"), "area") for window in WINDOWS]
    expected = torch.cat(rows).mean(dim=0, keepdim=True)
    document = model.document_logits(WINDOWS, "area")
  assert torch.allclose(single, padded)
  assert torch.allclose(document, expected)


def test_cpu_update_and_checkpoint_round_trip_preserve_probabilities(tmp_path: Path) -> None:
  model = tiny_model()
  sample = Sample("case", "pedido concedido", "F", "case", "Acórdão 1", "manual")
  example = TokenizedSample(sample, WINDOWS, target=0)
  before = next(model.encoder.parameters()).detach().clone()
  loss = train_step(model, example, build_optimizer(model))
  assert math.isfinite(loss)
  assert not torch.equal(before, next(model.encoder.parameters()))
  save_checkpoint(model, tmp_path / "checkpoint", tiny_tokenizer(), temperatures=(1.2, 1.3))
  expected = model.probabilities(WINDOWS, "area", temperature=1.2)
  loaded, _, temperatures = load_checkpoint(tmp_path / "checkpoint")
  actual = loaded.probabilities(WINDOWS, "area", temperature=temperatures[0])
  assert temperatures == (1.2, 1.3)
  assert actual == pytest.approx(expected)
  assert all(math.isfinite(value) for value in actual.values())
  assert sum(actual.values()) == pytest.approx(1.0)
  with sqlite3.connect(tmp_path / "checkpoint" / "checkpoint.db") as conn:
    conn.execute("UPDATE temperatures SET value=-1 WHERE task='area'")
  with pytest.raises(ValueError, match="invalid calibrated temperatures"):
    load_checkpoint(tmp_path / "checkpoint")
