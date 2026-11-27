"""Run Agathe: `python -m agavus.language_model agathe [civil] [wikipedia] [federal] [options]`.

Same command on a laptop and on a rented GPU (docs/llm-lab.md §6): there, the
text comes from an exported bundle (`--texts`) and `--out` keeps a run record
with config, seed, data hash, losses and timing, plus the weights.
"""

import argparse
import dataclasses
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from agavus.config.settings import load_settings
from agavus.language_model.agathe import AgatheConfig, AgatheModel, train
from agavus.language_model.bigram import build_vocab
from agavus.language_model.bundle import (
  Texts,
  bundle_bytes,
  digest,
  federal_texts,
  local_texts,
  read_bundle,
)
from agavus.language_model.data import Split, split_articles
from agavus.language_model.extra import without_leaks
from agavus.language_model.report import THREADS, chosen_steps, report
from agavus.language_model.training import synchronize

STEPS = 6000
CHECK_EVERY = 500
ATTENTION_TEXT = "Art. 22. Constatada a prática de violência doméstica"
SHOWN_LOOKS = 3
EXTRA_NAMES = ("civil", "wikipedia", "federal")


def parse(arguments: Sequence[str]) -> argparse.Namespace:
  """Options; every one has a CPU-friendly default."""
  parser = argparse.ArgumentParser(prog="python -m agavus.language_model agathe")
  parser.add_argument("extras", nargs="*", choices=EXTRA_NAMES, help="extra training text")
  parser.add_argument("--steps", type=int, default=STEPS, help="most steps to try on validation")
  parser.add_argument("--check-every", type=int, default=CHECK_EVERY)
  defaults = AgatheConfig()
  parser.add_argument("--layers", type=int, default=defaults.layers, help="transformer blocks")
  parser.add_argument("--width", type=int, default=defaults.width, help="embedding width")
  parser.add_argument("--heads", type=int, default=defaults.heads, help="attention heads per block")
  parser.add_argument("--context", type=int, default=defaults.context, help="tokens per window")
  parser.add_argument("--batch", type=int, default=defaults.batch, help="windows per update")
  parser.add_argument("--device", default="cpu", help="cpu, cuda, ...")
  parser.add_argument("--seed", type=int, default=0, help="starting weights, batches, dropout")
  parser.add_argument("--texts", type=Path, help="exported bundle instead of the local corpus")
  parser.add_argument("--out", type=Path, help="folder for run.json and model.pt")
  options = parser.parse_args(list(arguments))
  if min(options.layers, options.width, options.heads, options.context, options.batch) < 1:
    parser.error("layers, width, heads, context and batch must be positive integers")
  if options.width % options.heads:
    parser.error("--width must be divisible by --heads")
  return options


def show_attention(model: AgatheModel) -> None:
  """For the last letter, the earlier letters each head of the last layer looks at most."""
  looks = model.looks(ATTENTION_TEXT)[-1]
  letters = ["⟨start⟩", *ATTENTION_TEXT][-looks.shape[-1] :]
  print(f"text: {ATTENTION_TEXT!r}; who the last letter {letters[-1]!r} looks at:")
  for head, row in enumerate(looks[:, -1]):
    top = torch.topk(row, min(SHOWN_LOOKS, len(row)))
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
      print(f"lab: no {name} text in the selected inputs")
      return None
    chosen.extend(found)
  clean = without_leaks(chosen, texts.statutes)
  print(f"extra text: {len(clean)} texts, {sum(map(len, clean)):,} characters", end="")
  print(f" ({len(chosen) - len(clean)} dropped for overlapping a statute article)")
  return clean


def save_run(out: Path, model: AgatheModel, record: dict[str, object]) -> None:
  """run.json and model.pt, each written whole before it takes its final name."""
  out.mkdir(parents=True, exist_ok=True)
  weights = out / "model.pt.partial"
  torch.save([weight.detach().cpu() for weight in model.parameters()], weights)
  weights.rename(out / "model.pt")
  partial = out / "run.json.partial"
  partial.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
  partial.rename(out / "run.json")
  print(f"saved: {out / 'run.json'}, {out / 'model.pt'}")


def run_agathe(arguments: Sequence[str]) -> int:
  """Steps chosen on validation, exam opened once at the end.

  Extra text only joins the training side; validation, exam and the letter
  vocabulary stay the statute articles', so losses stay comparable.
  """
  options = parse(arguments)
  if options.texts:
    texts = read_bundle(options.texts)
  elif "federal" in options.extras:
    texts = federal_texts(load_settings().data_dir)
  else:
    texts = local_texts(load_settings().data_dir)
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
  config = AgatheConfig(
    layers=options.layers,
    width=options.width,
    heads=options.heads,
    context=options.context,
    batch=options.batch,
    check_every=options.check_every,
    device=options.device,
    seed=options.seed,
  )
  synchronize(config.device)
  began = time.monotonic()
  _, history = train(Split(inner.train + extra, inner.holdout), vocab, config, steps=options.steps)
  best = chosen_steps(history, len(history))
  model, _ = train(Split(split.train + extra, ()), vocab, config, steps=best)
  print(f"parameters: {model.size():,}")
  training, holdout, sample = report(model, split)
  show_attention(model)
  if options.out:
    synchronize(config.device)
    gpu = "mps" if torch.device(config.device).type == "mps" else None
    if torch.device(config.device).type == "cuda":
      gpu = torch.cuda.get_device_name(config.device)
    record: dict[str, object] = {
      "config": dataclasses.asdict(config),
      "parameters": model.size(),
      "extras": list(options.extras),
      "data_sha256": digest(options.texts)
      if options.texts
      else hashlib.sha256(bundle_bytes(texts)).hexdigest(),
      "gpu": gpu,
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
