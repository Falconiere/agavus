"""Tests for the training pipeline (tests folder, synthetic separable data)."""

from pathlib import Path

import pytest

from agavus.classify.features import VectorizerConfig
from agavus.classify.pipeline import train_classifier
from agavus.classify.store import load_model, save_model

TEXTS = [
  "medida protetiva violência doméstica ameaça",
  "medida protetiva descumprimento violência",
  "violência doméstica psicológica patrimonial",
  "protetiva urgência risco mulher",
  "guarda compartilhada filhos visitas",
  "divórcio partilha bens casal",
  "alimentos pensão filho menor",
  "guarda unilateral alienação parental",
]
LABELS = ["M", "M", "M", "M", "F", "F", "F", "F"]


def test_train_learns_and_reports() -> None:
  min_accuracy = 0.5
  tolerance = 1e-9
  classifier, report = train_classifier(
    TEXTS, LABELS, vectorizer_config=VectorizerConfig(min_df=1, char_sizes=())
  )
  assert report.n_train + report.n_holdout == len(TEXTS)
  assert report.holdout_accuracy >= min_accuracy
  probs = classifier.predict_proba(["violência doméstica protetiva"])
  assert abs(float(probs.sum()) - 1.0) < tolerance


def test_save_load_roundtrip(tmp_path: Path) -> None:
  classifier, report = train_classifier(
    TEXTS, LABELS, vectorizer_config=VectorizerConfig(min_df=1, char_sizes=())
  )
  db_path = tmp_path / "models.db"
  save_model(db_path, "classifier", classifier, report)
  loaded = load_model(db_path, "classifier")
  sample = ["guarda dos filhos", "medida protetiva"]
  assert (loaded.predict_proba(sample) == classifier.predict_proba(sample)).all()


def test_degenerate_inputs_raise() -> None:
  with pytest.raises(ValueError):
    train_classifier(["a"], ["F"])
  with pytest.raises(ValueError):
    train_classifier(["a", "b"], ["F", "F"])


def test_top_features_attributable() -> None:
  classifier, _ = train_classifier(
    TEXTS, LABELS, vectorizer_config=VectorizerConfig(min_df=1, char_sizes=())
  )
  features = classifier.top_features("violência doméstica protetiva")
  assert features
  assert all(isinstance(term, str) and isinstance(value, float) for term, value in features)
