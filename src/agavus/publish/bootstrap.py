"""Deterministic offline corpus bootstrap from bundled public-domain statutes.

Release builds and the public Space start from a clean data dir: the only
inputs are the public statute transcriptions in `space/assets/` plus authored
seeds. No network, no real case data (docs/publishing.md, LGPD).
"""

from pathlib import Path

from agavus.classify.pipeline import train_classifier, training_set
from agavus.classify.store import save_model
from agavus.config.settings import Settings
from agavus.corpus.ingest import import_statute_file
from agavus.corpus.records import TextSource
from agavus.corpus.storage import CorpusStorage

ASSET_NAMES: dict[str, str] = {
  "l11340": "Lei 11.340/2006",
  "l12318": "Lei 12.318/2010",
  "l13058": "Lei 13.058/2014",
  "l14994": "Lei 14.994/2024",
  "l10406": "Lei 10.406/2002",
}


def model_path(settings: Settings) -> Path:
  """Canonical location of the trained classifier artifact."""
  return settings.data_dir / "models" / "models.db"


def ensure_public_corpus(storage: CorpusStorage) -> None:
  """Reject release builds over non-public inputs (real cases stay private)."""
  if storage.load_cases():
    raise ValueError("data dir holds case records; release builds need a clean AGAVUS_DATA_DIR")
  foreign = sorted(
    document.origin_ref
    for document in storage.load_texts()
    if document.source is not TextSource.STATUTE
  )
  if foreign:
    raise ValueError(
      "data dir holds non-public texts; release builds need a clean "
      f"AGAVUS_DATA_DIR (first offender: {foreign[0]!r})"
    )


def bootstrap_corpus(settings: Settings, assets_dir: Path) -> Path:
  """Import bundled statutes, label, train; return the model db path.

  Idempotent: re-running replaces the model bundle and appends a training
  report. Refuses data dirs that already hold non-statute texts.
  """
  storage = CorpusStorage(settings.data_dir)
  ensure_public_corpus(storage)
  for stem, name in ASSET_NAMES.items():
    import_statute_file(settings, name, assets_dir / f"{stem}.html")
  storage.run_labeling()
  texts, targets = training_set(storage)
  classifier, report = train_classifier(texts, targets)
  save_model(model_path(settings), "classifier", classifier, report)
  return model_path(settings)


def ensure_model(settings: Settings, assets_dir: Path) -> Path:
  """Bootstrap once per data dir; reuse an already trained model."""
  path = model_path(settings)
  if not path.exists():
    bootstrap_corpus(settings, assets_dir)
  return path
