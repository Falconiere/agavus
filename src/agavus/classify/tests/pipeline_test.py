"""Tests for the training pipeline (tests folder, synthetic separable data)."""

from pathlib import Path

import pytest

from agavus.classify.features import VectorizerConfig
from agavus.classify.pipeline import TrainedClassifier, train_classifier

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
  classifier, _ = train_classifier(
    TEXTS, LABELS, vectorizer_config=VectorizerConfig(min_df=1, char_sizes=())
  )
  classifier.save(tmp_path)
  loaded = TrainedClassifier.load(tmp_path)
  sample = ["guarda dos filhos", "medida protetiva"]
  assert (loaded.predict_proba(sample) == classifier.predict_proba(sample)).all()


def test_degenerate_inputs_raise() -> None:
  with pytest.raises(ValueError):
    train_classifier(["a"], ["F"])
  with pytest.raises(ValueError):
    train_classifier(["a", "b"], ["F", "F"])
