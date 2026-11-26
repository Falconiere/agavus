# Private learned-classifier experiment

This is an optional comparison, permitted by DESIGN §6.1. It does not replace
production routing, retrieval weights or extractive answers. Install its libraries
with `uv sync --extra dev --extra learned`, then run from the repository root:

```sh
uv run --extra learned python -m agavus.classify.learned experiment \
  --device auto --budget-minutes 120
```

`AGAVUS_LEARNED_DEVICE` and `AGAVUS_LEARNED_BUDGET_MINUTES` provide the defaults.
`auto` tests a real local MPS training update; failure falls back to CPU. Explicit
CPU training and inference work. There are no rented machines or uploads.
The `dev` extra includes these libraries so `make check` exercises real CPU training;
an application installation without extras does not need them.

## Inputs and isolation

The command streams the existing corpus without relabeling it. Resolved labels
retain their source and citable document reference. Review flags exclude the whole
case, including sibling documents; the residual personal-data scan excludes flagged
documents/references. The snapshot adds exactly 100 distinct, eligible cached O
cases using the existing organ rules. Their labels are marked snapshot-derived and
weak; no additional F/M decisions enter this experiment.

The 347 existing statutes and 330 authored queries are training-only auxiliaries.
Actual eligible counts are recorded, not assumed. There are 120 authored relation
families, each with two support, two contradiction and two unknown pairs. The six
categories cover direct/double negation, passive wording, quotations, conditions and
multiple requests. Twenty shared wording templates link scenario variants across
categories; every family and template stays entirely within one partition.

Case identity, normalized duplicate text, family and template links are joined
transitively. Per-class dated case groups sort by latest publication, with stable
hash tie-breaking. Reserve 15% validation, 10% calibration and 15% final groups
(round down, at least one each); the remainder is training. Undated auxiliaries
cannot displace dated cases. Insufficient independent groups stop with a dataset
report. Class conflicts fail rather than being guessed.

## Learned computation

The trainable E5-small assets use the same pinned revision as the ONNX retrieval
encoder, in a separate experiment directory. Raw tokenizer IDs feed the encoder;
there is no synonym normalization, stopword filtering or handwritten negation,
decision-verb or citation feature extraction. Those features remain in TF-IDF.

For hidden token states `h` and mask `m`, pooling is
`z = sum(m * h) / sum(m)`. Separate linear heads compute area and internal
statement-relation logits. Area classes come from the registry; relation classes
are `support`, `contradiction`, `unknown`. No text or legal explanation is generated.

Documents use 480 payload-token windows and 64-token overlap. Every window has
`passage:` and boundary tokens. Statement pairs use `query: statement passage:
passage`; pairs over 480 tokens fail instead of being truncated. Inference averages
all window logits **before** temperature scaling.

Training averages window cross-entropies within each document. Class weights use
training counts only; each task has equal total weight. Gradients accumulate over
16 documents before AdamW updates, with encoder/head rates 2e-5/1e-3 and default
weight decay 0.01. Checkpoint selection minimizes validation NLL (equal task means
for joint training). Temperature fitting sees calibration only. No model learns
during inference.

Compare feature-preserving TF-IDF, a frozen E5 linear model, area-only E5 tuning
and joint tuning. Baselines are deterministic; both tuning variants use seeds
0, 1 and 2 with the largest common epoch count that the pilot estimates will fit,
capped at three. The pilot uses the same accumulation loop as training. Encoding,
pilot, training, calibration, CPU timing and evaluation share a computation ceiling;
evaluation is reserved before scheduling tuning. Deadlines are checked between
encoder windows, so the last bounded operation may cross the wall-time boundary.
An incomplete comparison has no winner.

## Artifacts and interpretation

Each unique ignored `data/experiments/learned/<run>/` contains:

- `experiment.db`: frozen examples, provenance, membership, versions, parameters,
  window counts, pilot/schedule, validation history, probabilities and results.
- `source.zip`: public source/build files, including uncommitted implementation,
  with source hashes in SQLite for replay alongside the frozen snapshot.
- Encoder/tokenizer assets, Safetensors heads and SQLite class/temperature metadata
  for each selected checkpoint; baseline weights remain separate from live models.
- `report.md`: final document macro-F1, M recall, unsafe decisions, ECE, accepted
  precision/coverage, relation macro-F1 and the conservative verdict. SQLite also
  records category failures, grant/denial flips with area stability, CPU document
  latency, process peak RSS, elapsed computation and seed variability.

Area-only relation-head scores are untrained-head diagnostics. Query development
results are not fresh gold. A promising candidate must beat both baselines, preserve
their M recall, have zero observed unsafe M decisions, ECE ≤ 0.05 and accepted
precision ≥ 0.95 across all seeds. A joint candidate must also improve relation F1
on unseen families. The separate M-recall target is 0.98. Weak case labels, a small
final partition and authored relation examples cannot establish production safety.
No experiment activates production or changes the M safety ceiling.
