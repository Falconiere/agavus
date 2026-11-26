"""Router: learns from vectors, persists, refits; synthetic encoder only."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from agavus.classify import router_store
from agavus.classify.router import TRAINING as ROUTER_TRAINING
from agavus.classify.router import QueryRouter, RouterWeights, calibration_folds, fit_router
from agavus.classify.router_store import ensure_router
from agavus.classify.seed_everyday import EVERYDAY_QUERIES
from agavus.classify.seed_scenarios import SCENARIOS
from agavus.evals.goldset import ROUTING_V1
from agavus.evals.routingset import DEV, HIDDEN
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


def test_scenarios_are_registered_unique_and_outside_evaluation_gold() -> None:
  """Catch duplicate scenario questions or accidental evaluation contamination."""
  questions = tuple(question for _, _, authored in SCENARIOS for question in authored)
  normalized = {question.casefold().strip(" ?.") for question in questions}
  gold = {item.text.casefold().strip(" ?.") for item in (*DEV, *HIDDEN, *ROUTING_V1)}

  assert len({scenario_id for scenario_id, _, _ in SCENARIOS}) == len(SCENARIOS)
  assert all(is_area(label) for _, label, _ in SCENARIOS)
  assert all(questions)
  assert len(normalized) == len(questions)
  assert not normalized & gold
  expected = {(question, label) for _, label, authored in SCENARIOS for question in authored}
  assert expected <= set(ROUTER_TRAINING)


def test_calibration_folds_keep_scenario_groups_together_and_leave_each_class_in_train() -> None:
  """Catch out-of-fold calibration leakage or a fold without a training class."""
  labels = tuple(label for label in "FMO" for _ in range(5) for _ in range(2))
  groups = tuple(f"{label}-{number}" for label in "FMO" for number in range(5) for _ in range(2))

  folds = calibration_folds(labels, groups)
  held = {index for fold in folds for index in fold}

  assert held == set(range(len(labels)))
  for group in set(groups):
    matching = [fold for fold in folds if any(groups[index] == group for index in fold)]
    assert len(matching) == 1
    assert {groups[index] for index in matching[0] if groups[index] == group} == {group}
  for fold in folds:
    kept = set(range(len(labels))) - set(fold)
    assert {labels[index] for index in kept} == set(labels)
