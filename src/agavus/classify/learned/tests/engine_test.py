"""Real CPU tests for raw-token learned-engine encoding and scoring."""

import time

import numpy as np

from agavus.classify.learned.engine import Execution, encode_samples, frozen_vectors, predict_logits
from agavus.classify.learned.types import Sample

from .fixtures import tiny_model, tiny_tokenizer


def test_engine_covers_long_documents_and_returns_one_finite_row_per_sample() -> None:
  samples = (
    Sample("long", "pedido " * 600, "F", "case-1", "ref-1", "manual"),
    Sample(
      "relation", "pedido negado", "support", "case-2", "ref-2", "synthetic", statement="pedido"
    ),
  )
  encoded = encode_samples(tiny_tokenizer(), samples)
  model = tiny_model()
  execution = Execution("cpu", time.monotonic() + 30)
  logits = predict_logits(model, encoded, execution)
  vectors = frozen_vectors(model, encoded, execution)
  assert len(encoded[0].windows) > 1
  assert logits.shape == (2, 3)
  assert vectors.shape == (2, 8)
  assert np.isfinite(logits).all() and np.isfinite(vectors).all()
