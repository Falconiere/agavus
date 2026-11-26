"""Tests for the training pipeline (tests folder, synthetic separable data)."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from agavus.classify.features import VectorizerConfig
from agavus.classify.pipeline import train_classifier, training_set
from agavus.classify.splits import TrainingData
from agavus.classify.store import load_model, save_model
from agavus.corpus.records import LabelSource, TextDocument, TextSource
from agavus.corpus.storage import CorpusStorage
from agavus.taxonomy.labeling import LabelDecision

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
EXAMPLES_PER_LABEL = 5
HOLDOUT_POSITION = EXAMPLES_PER_LABEL - 1


def test_train_learns_and_reports() -> None:
  min_accuracy = 0.5
  tolerance = 1e-9
  classifier, report = train_classifier(
    TEXTS, LABELS, vectorizer_config=VectorizerConfig(min_df=1, char_sizes=())
  )
  assert report.n_train + report.n_calibration + report.n_holdout == len(TEXTS)
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


def test_holdout_text_changes_do_not_reach_training_or_calibration() -> None:
  """Catch final-holdout leakage into vocabulary, weights, or temperature."""
  texts = tuple(
    f"{label} assunto {index}" for label in "FMO" for index in range(EXAMPLES_PER_LABEL)
  )
  labels = tuple(label for label in "FMO" for _ in range(EXAMPLES_PER_LABEL))
  groups = tuple(f"{label}-{index}" for label in "FMO" for index in range(EXAMPLES_PER_LABEL))
  dates = tuple(date(2024, 1, index + 1) for _ in "FMO" for index in range(EXAMPLES_PER_LABEL))
  changed = tuple(
    f"texto holdout isolado {label}" if index % EXAMPLES_PER_LABEL == HOLDOUT_POSITION else text
    for index, (text, label) in enumerate(zip(texts, labels, strict=True))
  )
  config = VectorizerConfig(min_df=1, char_sizes=())

  first, _ = train_classifier(TrainingData(texts, labels, groups, dates), vectorizer_config=config)
  second, _ = train_classifier(
    TrainingData(changed, labels, groups, dates), vectorizer_config=config
  )

  assert first.vectorizer.vocab == second.vectorizer.vocab
  assert np.array_equal(first.model.weights, second.model.weights)
  assert np.array_equal(first.model.bias, second.model.bias)
  assert first.temperature == second.temperature


def test_training_set_groups_cases_and_excludes_review_and_invalid_references(
  tmp_path: Path,
) -> None:
  """Catch case leakage and unsupported non-statute documents entering training."""
  storage = CorpusStorage(tmp_path)
  documents = _training_documents()
  assert storage.save_texts(documents) == len(documents)
  storage.save_labels(_training_decisions())

  data = training_set(storage)
  rows = dict(zip(data.texts, zip(data.labels, data.groups, data.dates, strict=True), strict=True))

  assert rows["acórdão um"] == ("M", "07012345620248070001", date(2024, 2, 1))
  assert rows["acórdão dois"] == ("M", "07012345620248070001", date(2024, 3, 1))
  assert rows["texto legal"] == ("M", "Lei 11.340/2006, art. 22", None)
  assert "em revisão" not in rows
  assert "sem processo" not in rows


def _training_documents() -> tuple[TextDocument, ...]:
  return (
    _document("case-masked", "acórdão um", "Processo 0701234-56.2024.8.07.0001", date(2024, 2, 1)),
    _document("case-plain", "acórdão dois", "Processo 07012345620248070001", date(2024, 3, 1)),
    _document("statute", "texto legal", "Lei 11.340/2006, art. 22", None),
    _document("review", "em revisão", "Processo 0709999-99.2024.8.07.0001", date(2024, 4, 1)),
    _document("invalid", "sem processo", "Acórdão sem número CNJ", date(2024, 5, 1)),
  )


def _training_decisions() -> list[LabelDecision]:
  return [
    LabelDecision(doc_id, "M", LabelSource.MANUAL, review, "synthetic")
    for doc_id, review in (
      ("case-masked", False),
      ("case-plain", False),
      ("statute", False),
      ("review", True),
      ("invalid", False),
    )
  ]


def _document(doc_id: str, text: str, origin_ref: str, published_at: date | None) -> TextDocument:
  return TextDocument(
    id=doc_id,
    source=TextSource.STATUTE if doc_id == "statute" else TextSource.PORTAL,
    origin_ref=origin_ref,
    published_at=published_at,
    text=text,
    citations=(),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="raw/synthetic.json",
  )
