"""Tests for the offline public-corpus bootstrap (synthetic assets only)."""

from pathlib import Path

import pytest

from agavus.corpus.httpfixtures import sample_case, sample_settings
from agavus.corpus.records import LabelSource, TextDocument, TextSource
from agavus.corpus.storage import CorpusStorage
from agavus.publish.bootstrap import ASSET_NAMES, bootstrap_corpus, ensure_model, model_path

FIXTURE_ARTICLES = {
  "l11340": "Art. 1º Esta lei cria mecanismos para coibir a violência doméstica.",
  "l12318": "Art. 2º Considera-se ato de alienação parental a interferência psicológica.",
  "l13058": "Art. 1º Esta lei dispõe sobre a guarda compartilhada dos filhos.",
  "l14994": "Art. 1º Esta lei agrava a pena do feminicídio no Código Penal.",
  "l10406": "Art. 1.511 O casamento estabelece comunhão plena de vida.",
}


def _assets(tmp_path: Path) -> Path:
  assets = tmp_path / "assets"
  assets.mkdir()
  for stem, body in FIXTURE_ARTICLES.items():
    (assets / f"{stem}.html").write_text(body, encoding="utf-8")
  return assets


def test_bootstrap_builds_public_model(tmp_path: Path) -> None:
  settings = sample_settings(tmp_path / "data")
  db_path = bootstrap_corpus(settings, _assets(tmp_path))
  assert db_path == model_path(settings)
  assert db_path.exists()
  documents = CorpusStorage(settings.data_dir).load_texts()
  assert len(documents) == len(ASSET_NAMES)
  assert {document.source for document in documents} == {TextSource.STATUTE}


def test_bootstrap_refuses_non_public_texts(tmp_path: Path) -> None:
  settings = sample_settings(tmp_path / "data")
  storage = CorpusStorage(settings.data_dir)
  storage.save_text(
    TextDocument(
      id="d1",
      source=TextSource.DJEN,
      origin_ref="DJEN decisão sintética",
      published_at=None,
      text="texto sintético",
      citations=(),
      domain_label=None,
      label_source=LabelSource.PENDING,
      raw_ref="raw/d1.json",
    )
  )
  with pytest.raises(ValueError, match="non-public"):
    bootstrap_corpus(settings, _assets(tmp_path))
  assert not model_path(settings).exists()


def test_bootstrap_refuses_case_records(tmp_path: Path) -> None:
  settings = sample_settings(tmp_path / "data")
  storage = CorpusStorage(settings.data_dir)
  storage.save_case(sample_case())
  with pytest.raises(ValueError, match="case records"):
    bootstrap_corpus(settings, _assets(tmp_path))
  assert not model_path(settings).exists()


def test_ensure_model_reuses_existing_model(tmp_path: Path) -> None:
  settings = sample_settings(tmp_path / "data")
  db_path = bootstrap_corpus(settings, _assets(tmp_path))
  assert ensure_model(settings, tmp_path / "missing-assets") == db_path
