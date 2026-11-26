"""Run the tiny GPT: `python -m agavus.lab gpt [civil] [wikipedia] [options]`.

Same command on a laptop and on a rented GPU (docs/llm-lab.md §6): there, the
text comes from an exported bundle (`--texts`) and `--out` keeps a run record
with config, seed, data hash, losses and timing, plus the weights.
"""

import argparse
import dataclasses
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from agavus.config.settings import load_settings
from agavus.lab.bigram import build_vocab
from agavus.lab.bundle import Texts, digest, local_texts, read_bundle
from agavus.lab.data import Split, split_articles
from agavus.lab.extra import without_leaks
from agavus.lab.gpt import GptConfig, GptModel, train
from agavus.lab.report import THREADS, chosen_steps, report

STEPS = 6000
CHECK_EVERY = 500
ATTENTION_TEXT = "Art. 22. Constatada a prática de violência doméstica"
SHOWN_LOOKS = 3
EXTRA_NAMES = ("civil", "wikipedia")


def parse(arguments: Sequence[str]) -> argparse.Namespace:
  """Options; every one has a CPU-friendly default."""
  parser = argparse.ArgumentParser(prog="python -m agavus.lab gpt")
  parser.add_argument("extras", nargs="*", choices=EXTRA_NAMES, help="extra training text")
  parser.add_argument("--steps", type=int, default=STEPS, help="most steps to try on validation")
  parser.add_argument("--check-every", type=int, default=CHECK_EVERY)
  parser.add_argument("--device", default="cpu", help="cpu, cuda, ...")
  parser.add_argument("--texts", type=Path, help="exported bundle instead of the local corpus")
  parser.add_argument("--out", type=Path, help="folder for run.json and model.pt")
  return parser.parse_args(list(arguments))


def show_attention(model: GptModel) -> None:
  """For the last letter, the earlier letters each head of the last layer looks at most."""
  looks = model.looks(ATTENTION_TEXT)[-1]
  letters = ["⟨start⟩", *ATTENTION_TEXT][-looks.shape[-1] :]
  print(f"text: {ATTENTION_TEXT!r}; who the last letter {letters[-1]!r} looks at:")
  for head, row in enumerate(looks[:, -1]):
    top = torch.topk(row, SHOWN_LOOKS)
    shown = ", ".join(
      f"{letters[int(where)]!r} at {int(where)} ({float(share):.0%})"
      for share, where in zip(top.values, top.indices, strict=True)
    )
    print(f"  head {head}: {shown}")


def extra_texts(texts: Texts, names: Sequence[str]) -> tuple[str, ...] | None:
  """Requested extra text, minus anything that overlaps a statute article."""
  chosen: list[str] = []
  for name in names:
    found = texts.extras.get(name, ())
    if not found:
      print(f"lab: no {name} text found (wikipedia needs: python -m agavus.lab fetch-wikipedia)")
      return None
    chosen.extend(found)
  clean = without_leaks(chosen, texts.statutes)
  print(f"extra text: {len(clean)} texts, {sum(map(len, clean)):,} characters", end="")
  print(f" ({len(chosen) - len(clean)} dropped for overlapping a statute article)")
  return clean


def save_run(out: Path, model: GptModel, record: dict[str, object]) -> None:
  """run.json and model.pt, each written whole before it takes its final name."""
  out.mkdir(parents=True, exist_ok=True)
  weights = out / "model.pt.partial"
  torch.save([weight.detach().cpu() for weight in model.parameters()], weights)
  weights.rename(out / "model.pt")
  partial = out / "run.json.partial"
  partial.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
  partial.rename(out / "run.json")
  print(f"saved: {out / 'run.json'}, {out / 'model.pt'}")


def run_gpt(arguments: Sequence[str]) -> int:
  """Steps chosen on validation, exam opened once at the end.

  Extra text only joins the training side; validation, exam and the letter
  vocabulary stay the statute articles', so losses stay comparable.
  """
  options = parse(arguments)
  texts = read_bundle(options.texts) if options.texts else local_texts(load_settings().data_dir)
  if not texts.statutes:
    print("lab: no statute articles — ingest statutes first")
    return 1
  extra = extra_texts(texts, options.extras)
  if extra is None:
    return 1
  torch.set_num_threads(THREADS)
  split = split_articles(texts.statutes)
  inner = split_articles(split.train)
  vocab = build_vocab(split.train)
  config = GptConfig(check_every=options.check_every, device=options.device)
  began = time.monotonic()
  _, history = train(Split(inner.train + extra, inner.holdout), vocab, config, steps=options.steps)
  best = chosen_steps(history, len(history))
  model, _ = train(Split(split.train + extra, ()), vocab, config, steps=best)
  print(f"parameters: {model.size():,}")
  training, holdout, sample = report(model, split)
  show_attention(model)
  if options.out:
    record: dict[str, object] = {
      "config": dataclasses.asdict(config),
      "extras": list(options.extras),
      "data_sha256": digest(options.texts) if options.texts else "local corpus",
      "gpu": torch.cuda.get_device_name() if options.device.startswith("cuda") else None,
      "torch": torch.__version__,
      "steps_tried": options.steps,
      "steps_chosen": best,
      "history": history,
      "loss_train": training,
      "loss_holdout": holdout,
      "sample": sample,
      "seconds": round(time.monotonic() - began),
    }
    save_run(options.out, model, record)
  return 0
