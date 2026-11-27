"""Agathe runner: options, extra text, and a tiny end-to-end run (synthetic text)."""

import json
from pathlib import Path

import pytest
import torch

from agavus.model.agathe import AgatheConfig, start
from agavus.model.agathe_run import extra_texts, parse, run_agathe
from agavus.model.bigram import build_vocab
from agavus.model.bundle import Texts, digest, write_bundle
from agavus.model.data import split_articles

# Many threads only fight over work this small.
torch.set_num_threads(1)

STATUTES = tuple(f"Art. {n}º A guarda do filho {n} será compartilhada." for n in range(1, 26))
TEXTS = Texts(statutes=STATUTES, extras={"civil": ("Art. 900. A posse.",), "wikipedia": ()})
BASELINE_LAYERS = 4
FIVE_LAYERS = 5
RATE = 1e-3


def test_defaults_run_on_cpu() -> None:
  options = parse([])
  assert (options.device, options.extras, options.texts, options.out) == ("cpu", [], None, None)
  assert options.seed == 0
  assert options.layers == BASELINE_LAYERS


@pytest.mark.parametrize("flag", ["--layers", "--width", "--heads", "--context", "--batch"])
def test_zero_dimension_is_refused(flag: str) -> None:
  with pytest.raises(SystemExit):
    parse([flag, "0"])


@pytest.mark.parametrize("rate", ["0", "-1e-3", "nan"])
def test_non_positive_rate_is_refused(rate: str) -> None:
  with pytest.raises(SystemExit):
    parse(["--rate", rate])


def test_width_must_divide_into_heads() -> None:
  with pytest.raises(SystemExit):
    parse(["--width", "127", "--heads", "4"])


def test_unknown_extra_is_refused() -> None:
  with pytest.raises(SystemExit):
    parse(["novels"])


def test_extra_texts_refuses_missing_text() -> None:
  assert extra_texts(TEXTS, ["civil"]) == ("Art. 900. A posse.",)
  assert extra_texts(TEXTS, ["wikipedia"]) is None


def test_extra_texts_drop_leaks() -> None:
  leaked = Texts(statutes=("x" * 200,), extras={"civil": ("y " + "x" * 150, "ok")})
  assert extra_texts(leaked, ["civil"]) == ("ok",)


def _run(bundle: Path, out: Path, arguments: list[str]) -> str:
  """Train from a synthetic bundle and return the saved run record's JSON."""
  write_bundle(bundle, TEXTS)
  assert run_agathe([*arguments, "--texts", str(bundle), "--out", str(out)]) == 0
  return (out / "run.json").read_text(encoding="utf-8")


def test_run_from_bundle_saves_record_and_weights(tmp_path: Path) -> None:
  bundle = tmp_path / "texts.json"
  arguments = ["civil", "--steps", "2", "--check-every", "1", "--seed", "3"]
  shape = ["--layers", "5", "--width", "12", "--heads", "3", "--context", "2", "--batch", "8"]
  record = json.loads(_run(bundle, tmp_path / "run", [*arguments, *shape, "--rate", "1e-3"]))
  out = tmp_path / "run"
  assert record["data_sha256"] == digest(bundle)
  assert record["steps_chosen"] in {1, 2}
  assert (record["config"]["device"], record["config"]["seed"]) == ("cpu", len("abc"))
  assert len(record["history"]) == len("12")
  assert record["config"]["layers"] == FIVE_LAYERS
  config = AgatheConfig(layers=5, width=12, heads=3, context=2, batch=8)
  expected = start(build_vocab(split_articles(STATUTES).train), config)
  assert record["parameters"] == expected.size()
  assert record["config"]["batch"] == config.batch
  assert record["config"]["rate"] == RATE
  assert [w.shape for w in torch.load(out / "model.pt")] == [w.shape for w in expected.parameters()]
  assert not list(out.glob("*.partial"))


def test_once_keeps_the_last_step_and_accepts_local_texts(tmp_path: Path) -> None:
  steps = ["civil", "--steps", "3", "--check-every", "1", "--once"]
  shape = ["--width", "12", "--heads", "3", "--context", "2", "--batch", "4"]
  record = json.loads(_run(tmp_path / "texts.json", tmp_path / "once", [*steps, *shape]))
  assert (record["steps_tried"], record["steps_chosen"]) == (len("abc"), len("abc"))
  assert parse(["djen", "planalto"]).extras == ["djen", "planalto"]
