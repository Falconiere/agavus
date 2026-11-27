"""Lab text bundle: round trip and hash (synthetic text)."""

import json
from pathlib import Path

from agavus.corpus.storage import CorpusStorage
from agavus.model.bundle import (
  BUNDLE_FILE,
  Texts,
  digest,
  local_texts,
  read_bundle,
  with_local,
  write_bundle,
)
from agavus.model.planalto import PLANALTO_DIR


def test_bundle_round_trip_and_hash(tmp_path: Path) -> None:
  texts = Texts(statutes=("Art. 1º Toda pessoa.",), extras={"civil": ("Art. 2º A lei.",)})
  path = tmp_path / "out" / "texts.json"
  sha = write_bundle(path, texts)
  assert read_bundle(path) == texts
  assert sha == digest(path)
  assert len(sha) == len("0" * 64)


def test_same_texts_give_the_same_hash(tmp_path: Path) -> None:
  texts = Texts(statutes=("a", "b"), extras={"wikipedia": ("c",), "civil": ()})
  assert write_bundle(tmp_path / "1.json", texts) == write_bundle(tmp_path / "2.json", texts)


def test_local_texts_without_extras(tmp_path: Path) -> None:
  CorpusStorage(tmp_path).save_texts([])
  texts = local_texts(tmp_path)
  assert texts.statutes == ()
  assert texts.extras == {"civil": (), "wikipedia": ()}


def test_with_local_adds_named_texts_to_the_frozen_bundle(tmp_path: Path) -> None:
  base = Texts(("Art. 1º Lei.",), {"civil": ("Art. 900.",)})
  write_bundle(tmp_path / BUNDLE_FILE, base)
  folder = tmp_path / PLANALTO_DIR / "articles" / "leis"
  folder.mkdir(parents=True)
  (folder / "l1.htm.jsonl").write_text(json.dumps({"text": "Art. 2º Outra."}) + "\n")
  texts = with_local(tmp_path, ["civil", "planalto"])
  assert texts.statutes == base.statutes
  assert texts.extras == {"civil": ("Art. 900.",), "planalto": ("Art. 2º Outra.",)}
