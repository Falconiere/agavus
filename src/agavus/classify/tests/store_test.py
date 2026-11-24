"""Tests for the SQLite model artifact store (tests folder)."""

from pathlib import Path

import pytest

from agavus.classify import store
from agavus.classify.features import VectorizerConfig
from agavus.classify.pipeline import train_classifier

TEXTS = ["protetiva violência", "protetiva ameaça", "guarda filhos", "divórcio bens"]
LABELS = ["M", "M", "F", "F"]


def test_missing_model_raises(tmp_path: Path) -> None:
  with pytest.raises(KeyError):
    store.load_model(tmp_path / "models.db", "classifier")


def test_replace_keeps_single_bundle(tmp_path: Path) -> None:
  db_path = tmp_path / "models.db"
  classifier, report = train_classifier(
    TEXTS, LABELS, vectorizer_config=VectorizerConfig(min_df=1, char_sizes=())
  )
  store.save_model(db_path, "classifier", classifier, report)
  store.save_model(db_path, "classifier", classifier, report)
  loaded = store.load_model(db_path, "classifier")
  assert loaded.model.classes == classifier.model.classes
  assert len(loaded.vectorizer.vocab) == len(classifier.vectorizer.vocab)
