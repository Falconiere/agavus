"""GPT runner: options, extra text, and a tiny end-to-end run (synthetic text)."""

import json
from pathlib import Path

import pytest
import torch

from agavus.lab.bundle import Texts, digest, write_bundle
from agavus.lab.gpt_run import extra_texts, parse, run_gpt

# Many threads only fight over work this small.
torch.set_num_threads(1)

STATUTES = tuple(f"Art. {n}º A guarda do filho {n} será compartilhada." for n in range(1, 26))
TEXTS = Texts(statutes=STATUTES, extras={"civil": ("Art. 900. A posse.",), "wikipedia": ()})


def test_defaults_run_on_cpu() -> None:
  options = parse([])
  assert (options.device, options.extras, options.texts, options.out) == ("cpu", [], None, None)


def test_unknown_extra_is_refused() -> None:
  with pytest.raises(SystemExit):
    parse(["novels"])


def test_extra_texts_refuses_missing_text() -> None:
  assert extra_texts(TEXTS, ["civil"]) == ("Art. 900. A posse.",)
  assert extra_texts(TEXTS, ["wikipedia"]) is None


def test_extra_texts_drop_leaks() -> None:
  leaked = Texts(statutes=("x" * 200,), extras={"civil": ("y " + "x" * 150, "ok")})
  assert extra_texts(leaked, ["civil"]) == ("ok",)


def test_run_from_bundle_saves_record_and_weights(tmp_path: Path) -> None:
  bundle = tmp_path / "texts.json"
  write_bundle(bundle, TEXTS)
  out = tmp_path / "run"
  arguments = ["civil", "--steps", "2", "--check-every", "1", "--texts", str(bundle)]
  assert run_gpt([*arguments, "--out", str(out)]) == 0
  record = json.loads((out / "run.json").read_text(encoding="utf-8"))
  assert record["data_sha256"] == digest(bundle)
  assert record["steps_chosen"] in {1, 2}
  assert record["config"]["device"] == "cpu"
  assert len(record["history"]) == len("12")
  assert len(torch.load(out / "model.pt")) > 0
  assert not list(out.glob("*.partial"))
