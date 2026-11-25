"""Router: learns from vectors, persists, refits; synthetic encoder only."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from agavus.classify import router_store
from agavus.classify.router import QueryRouter, RouterWeights, fit_router
from agavus.classify.router_store import ensure_router
from agavus.classify.seed_everyday import EVERYDAY_QUERIES
from agavus.retrieve.encoder import Encoder, FloatArray
from agavus.taxonomy.areas import is_area

TRAINING = tuple((f"{area} {number}", area) for area in ("F", "M", "O") for number in range(10))


class AxisEncoder:
  """Unit vector chosen by the first letter."""

  def encode_queries(self, texts: Sequence[str]) -> FloatArray:
    """One axis per area letter."""
    return np.array([np.eye(3)["FMO".index(text[0])] for text in texts])

  def encode_passages(self, texts: Sequence[str]) -> FloatArray:
    """Same as queries."""
    return self.encode_queries(texts)


def test_router_learns_and_is_normalized() -> None:
  router = QueryRouter(AxisEncoder(), fit_router(AxisEncoder(), TRAINING))
  for area, mapping in zip("FMO", router.probabilities(["F x", "M x", "O x"]), strict=True):
    assert max(mapping, key=lambda key: mapping[key]) == area
    assert np.isclose(sum(mapping.values()), 1.0)


def test_fit_is_deterministic() -> None:
  first = fit_router(AxisEncoder(), TRAINING)
  second = fit_router(AxisEncoder(), TRAINING)
  assert np.array_equal(first.model.weights, second.model.weights)
  assert first.temperature == second.temperature


def test_router_is_stored_and_refit_when_stale(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  path = tmp_path / "models.db"
  fits: list[int] = []

  def fit(encoder: Encoder) -> RouterWeights:
    fits.append(1)
    return fit_router(encoder, TRAINING)

  monkeypatch.setattr(router_store, "fit_router", fit)
  built = ensure_router(path, AxisEncoder()).probabilities(["M x"])
  assert ensure_router(path, AxisEncoder()).probabilities(["M x"]) == built
  assert len(fits) == 1
  monkeypatch.setattr(router_store, "fingerprint", lambda: "changed")
  ensure_router(path, AxisEncoder())
  assert len(fits) > 1


def test_everyday_seeds_are_valid_and_unique() -> None:
  assert all(is_area(label) for _, label in EVERYDAY_QUERIES)
  assert len({text for text, _ in EVERYDAY_QUERIES}) == len(EVERYDAY_QUERIES)
