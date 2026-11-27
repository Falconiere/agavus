"""Run Agathe: `python -m agavus.model agathe [civil] [wikipedia] [federal] [options]`.

Same command on a laptop and on a rented GPU (docs/llm-lab.md §6): there, the
text comes from an exported bundle (`--texts`) and `--out` keeps a run record
with config, seed, data and tokenizer hashes, losses and timing, plus the weights.
Validation and exam are reported in bits per character, comparable across tokenizers.
"""

import argparse
import dataclasses
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from tokenizers import Tokenizer

from agavus.config.settings import load_settings
from agavus.model.agathe import AgatheConfig, AgatheModel, check_config, train
from agavus.model.bundle import (
  LOCAL,
  Texts,
  bundle_bytes,
  digest,
  local_texts,
  read_bundle,
  with_local,
)
from agavus.model.data import Split, split_articles
from agavus.model.extra import without_leaks
from agavus.model.report import THREADS, chosen_steps, report
from agavus.model.tokenizer import TOKENIZER_FILE, decode, encode, load_tokenizer
from agavus.model.training import synchronize

STEPS = 6000
CHECK_EVERY = 500
ATTENTION_TEXT = "Art. 22. Constatada a prática de violência doméstica"
SHOWN_LOOKS = 3
EXTRA_NAMES = ("civil", "wikipedia", "federal", "planalto", "djen")


def parse(arguments: Sequence[str]) -> argparse.Namespace:
  """Options; the defaults are the GPU-sized model from docs/llm-lab.md §9."""
  parser = argparse.ArgumentParser(prog="python -m agavus.model agathe")
  parser.add_argument("extras", nargs="*", choices=EXTRA_NAMES, help="extra training text")
  parser.add_argument("--steps", type=int, default=STEPS, help="most steps to try on validation")
  parser.add_argument("--check-every", type=int, default=CHECK_EVERY)
  defaults = AgatheConfig()
  parser.add_argument("--layers", type=int, default=defaults.layers, help="transformer blocks")
  parser.add_argument("--width", type=int, default=defaults.width, help="embedding width")
  parser.add_argument("--heads", type=int, default=defaults.heads, help="attention heads per block")
  parser.add_argument(
    "--kv-heads", type=int, default=defaults.kv_heads, help="key/value heads shared by queries"
  )
  parser.add_argument("--rope-base", type=float, default=defaults.rope_base, help="RoPE base")
  parser.add_argument("--context", type=int, default=defaults.context, help="tokens per window")
  parser.add_argument("--batch", type=int, default=defaults.batch, help="windows per update")
  parser.add_argument("--rate", type=float, default=defaults.rate, help="AdamW learning rate")
  parser.add_argument("--device", default="cpu", help="cpu, cuda, ...")
  parser.add_argument("--seed", type=int, default=0, help="starting weights, batches, dropout")
  parser.add_argument("--bf16", action="store_true", help="bfloat16 mixed precision")
  parser.add_argument(
    "--once", action="store_true", help="train once and keep the last step (large text)"
  )
  parser.add_argument("--texts", type=Path, help="exported bundle instead of the local corpus")
  parser.add_argument("--out", type=Path, help="folder for run.json and model.pt")
  parser.add_argument(
    "--tokenizer", type=Path, help=f"BPE tokenizer file (default: data dir / {TOKENIZER_FILE})"
  )
  options = parser.parse_args(list(arguments))
  if options.batch < 1:
    parser.error("--batch must be a positive integer")
  if not options.rate > 0 or not options.rope_base > 0:
    parser.error("--rate and --rope-base must be positive")
  try:
    check_config(config_from(options))
  except ValueError as error:
    parser.error(str(error))
  return options


def config_from(options: argparse.Namespace) -> AgatheConfig:
  """The model and training settings the options ask for."""
  return AgatheConfig(
    layers=options.layers,
    width=options.width,
    heads=options.heads,
    kv_heads=options.kv_heads,
    rope_base=options.rope_base,
    context=options.context,
    batch=options.batch,
    rate=options.rate,
    bf16=options.bf16,
    check_every=options.check_every,
    device=options.device,
    seed=options.seed,
  )


def show_attention(model: AgatheModel) -> None:
  """For the last token, the earlier tokens each head of the last layer looks at most."""
  looks = model.looks(ATTENTION_TEXT)[-1]
  pieces = [decode(model.tokenizer, [token]) for token in encode(model.tokenizer, ATTENTION_TEXT)]
  letters = ["⟨start⟩", *pieces][-looks.shape[-1] :]
  print(f"text: {ATTENTION_TEXT!r}; who the last token {letters[-1]!r} looks at:")
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


def _inputs(options: argparse.Namespace) -> tuple[Texts, Tokenizer, Path] | None:
  """The texts and tokenizer the options name, or None after saying what is missing."""
  data_dir = load_settings().data_dir
  if options.texts:
    texts = read_bundle(options.texts)
  elif set(options.extras) & LOCAL.keys():
    texts = with_local(data_dir, options.extras)
  else:
    texts = local_texts(data_dir)
  if not texts.statutes:
    print("lab: no statute articles — ingest statutes first")
    return None
  path: Path = options.tokenizer or data_dir / TOKENIZER_FILE
  if not path.is_file():
    print(f"lab: no tokenizer at {path} — run `python -m agavus.model train-tokenizer` first")
    return None
  return texts, load_tokenizer(path), path


def _bits(model: AgatheModel, name: str, texts: Sequence[str]) -> float:
  """Print and return bits per character on `texts`."""
  bits = model.bits_per_char(texts)
  print(f"bits/char {name}: {bits:.4f}")
  return bits


def _gpu(device: str) -> str | None:
  """The accelerator a run used, for the record."""
  kind = torch.device(device).type
  if kind == "cuda":
    return torch.cuda.get_device_name(device)
  return "mps" if kind == "mps" else None


def run_agathe(arguments: Sequence[str]) -> int:
  """Steps chosen on validation, exam opened once at the end.

  Extra text only joins the training side; validation and exam stay the statute
  articles', so bits per character stay comparable.
  """
  options = parse(arguments)
  found = _inputs(options)
  if found is None:
    return 1
  texts, tokenizer, tokenizer_path = found
  extra = extra_texts(texts, options.extras)
  if extra is None:
    return 1
  torch.set_num_threads(THREADS)
  split = split_articles(texts.statutes)
  inner = split_articles(split.train)
  config = config_from(options)
  synchronize(config.device)
  began = time.monotonic()
  model, history = train(
    Split(inner.train + extra, inner.holdout), tokenizer, config, steps=options.steps
  )
  validation = _bits(model, "validation", inner.holdout)
  best = chosen_steps(history, len(history))
  if options.once:
    if history and best != options.steps:
      print(f"lab: validation was lowest at step {best}; keeping step {options.steps} (--once)")
    best = options.steps
  else:
    model, _ = train(Split(split.train + extra, ()), tokenizer, config, steps=best)
  print(f"parameters: {model.size():,}")
  training, holdout, sample = report(model, split)
  exam = _bits(model, "exam", split.holdout)
  show_attention(model)
  if options.out:
    synchronize(config.device)
    record: dict[str, object] = {
      "config": dataclasses.asdict(config),
      "parameters": model.size(),
      "extras": list(options.extras),
      "data_sha256": digest(options.texts)
      if options.texts
      else hashlib.sha256(bundle_bytes(texts)).hexdigest(),
      "tokenizer_sha256": digest(tokenizer_path),
      "gpu": _gpu(config.device),
      "torch": torch.__version__,
      "steps_tried": options.steps,
      "steps_chosen": best,
      "history": history,
      "loss_train": training,
      "loss_holdout": holdout,
      "bits_per_char_validation": validation,
      "bits_per_char_holdout": exam,
      "sample": sample,
      "seconds": round(time.monotonic() - began),
    }
    save_run(options.out, model, record)
  return 0
