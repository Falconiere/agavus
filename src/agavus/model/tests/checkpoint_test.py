"""Restore or reject real checkpoint files generated with a synthetic text bundle."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from agavus.model.agathe import AgatheConfig, start
from agavus.model.agathe_run import save_run
from agavus.model.bigram import build_vocab
from agavus.model.bundle import Texts, bundle_bytes, federal_texts, read_bundle, write_bundle
from agavus.model.checkpoint import load_checkpoint
from agavus.model.federal import LAWS
from agavus.model.report import SAMPLE_CHARS, SAMPLE_SEED
from agavus.model.tests.federal_test import write_federal_inputs


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, Path]:
  run, bundle = tmp_path / "run", tmp_path / "texts.json"
  sha = write_bundle(bundle, Texts(("only held out", "abc", "bca")))
  model = start(build_vocab(["abc"]), AgatheConfig(context=4, width=8, heads=2, layers=1))
  sample = model.sample(seed=SAMPLE_SEED, max_chars=SAMPLE_CHARS)
  record: dict[str, object] = {
    "config": {**asdict(model.config), "device": "cuda"},
    "data_sha256": sha,
    "sample": sample,
  }
  save_run(run, model, record)
  return run, bundle


def test_load_reconstructs_training_vocab_and_replays_saved_sample(
  checkpoint: tuple[Path, Path],
) -> None:
  run, bundle = checkpoint
  loaded = load_checkpoint(run, bundle)
  assert all(p.device.type == "cpu" for p in loaded.parameters())
  assert loaded.vocab == build_vocab(["abc"])
  sample = json.loads((run / "run.json").read_bytes())["sample"]
  assert loaded.sample(seed=SAMPLE_SEED, max_chars=SAMPLE_CHARS) == sample


@pytest.mark.parametrize("artifact", ["texts.json", "model.pt", "run.json"])
def test_reports_the_incompatible_artifact(checkpoint: tuple[Path, Path], artifact: str) -> None:
  run, bundle = checkpoint
  if artifact == "model.pt":
    weights = torch.load(run / artifact, weights_only=True)
    weights[0] = weights[0][0]
    torch.save(weights, run / artifact)
  elif artifact == "texts.json":
    write_bundle(bundle, Texts(("changed",)))
  else:
    (run / artifact).write_text("{}")
  with pytest.raises(ValueError, match=artifact):
    load_checkpoint(run, bundle)


def test_corrupt_weight_file_names_the_checkpoint(checkpoint: tuple[Path, Path]) -> None:
  run, bundle = checkpoint
  (run / "model.pt").write_bytes(b"bad")
  with pytest.raises(ValueError, match=r"model\.pt"):
    load_checkpoint(run, bundle)


def test_federal_run_checks_the_bundle_plus_local_federal_text(
  checkpoint: tuple[Path, Path], tmp_path: Path
) -> None:
  run, bundle = checkpoint
  folder = write_federal_inputs(tmp_path, read_bundle(bundle), "Art. 1º Lei.")
  record = json.loads((run / "run.json").read_bytes())
  record["extras"] = ["civil", "federal"]
  record["data_sha256"] = hashlib.sha256(bundle_bytes(federal_texts(tmp_path))).hexdigest()
  (run / "run.json").write_text(json.dumps(record))
  assert load_checkpoint(run, bundle, tmp_path).vocab == build_vocab(["abc"])
  with pytest.raises(ValueError, match="SHA-256"):
    load_checkpoint(run, bundle)
  (folder / f"{LAWS[0].stem}.jsonl").write_text('{"text":"Art. 2º Outra."}\n', encoding="utf-8")
  with pytest.raises(ValueError, match="SHA-256"):
    load_checkpoint(run, bundle, tmp_path)
