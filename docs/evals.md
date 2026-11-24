# Evals — guaranteeing and benchmarking quality

Status: design. Principle: **no change ships without measured evidence.**
Gold sets are the constitution; the harness is the gate. Future module:
`evals/` (cross-cutting, active from the `classify` step onward).

## Quality dimensions, metrics, targets

Targets are placeholders to calibrate on first real data — but the **hard**
rows are non-negotiable from day one.

| Dimension | Metric | Target |
| ------------------ | -------------------------------------- | ---------------------- |
| Routing | macro-F1 per area (held-out, time split) | ≥ 0.90 |
| Routing safety | **M recall** | ≥ 0.98 |
| Routing safety | M→F errors on safety-set | **0 (hard)** |
| Calibration | ECE + reliability diagram | ≤ 0.05 |
| Decision | precision on accepted answers | ≥ 0.95 |
| Decision | coverage (accepted / total) | reported, tuned via α/β |
| Retrieval | recall@5 on gold questions | ≥ 0.90 |
| Answers | citation correctness (right law, article, text) | **1.00 (hard)** |
| Answers | groundedness (every claim span-supported) | **1.00 (hard)** |
| Scope | OOD rejection rate | ≥ 0.95 |
| Privacy | party-name leakage scan | **0 (hard)** |
| Tiny LLM (later) | hallucination on gold set | **0 (hard)** |
| Tiny LLM (later) | verifier fallback rate, p95 latency | bounded, tracked |
| Reproducibility | same input → same output | exact (classical core) |

Coverage is always reported **next to** precision — the accuracy/coverage
dial is a product decision, not a hidden trade-off.

## Gold sets (versioned, firewalled)

| Set | Content | Lives |
| --------------- | ---------------------------------------------- | -------------- |
| routing-set | texts/queries + true area, incl. hard M↔F cases | git (authored) |
| retrieval-set | questions → relevant provisions (article-level judgments) | git (authored) |
| answer-set | questions → expected decision + citation + spans | git (authored) |
| safety-set | adversarial: M phrased as family dispute, urgency probes, OOD, injection texts (LLM layer) | git (authored) |
| calibration-holdout | labeled real cases, time-split | **outside git** |

Rules:

1. **Firewall**: gold data never enters training. Violation invalidates
   every metric.
2. Authored sets contain no personal data → safe in the public repo.
   Case-derived holdouts follow the corpus policy (gitignored).
3. Sets are versioned; every report names the set versions used.
4. **Growth loop**: reviewed boundary cases flow into gold sets (active
   learning, most-uncertain first) — the defer queue funds the benchmark.

## Splits and leakage

- Train/held-out split **by case, by time** (train ≤ 2025, eval 2026+):
  evaluation mimics future cases.
- Dedup before splitting: same case/acórdão never on both sides.
- Vocabulary, IDF, model, calibrator fitted on train only.

## Harness and regression policy

- `make eval` runs all dimensions → JSON + Markdown report (metrics, set
  versions, corpus/model versions, confusion matrix, reliability diagram,
  accuracy/coverage curve).
- **Baselines committed** (metrics only, no data): a change that regresses
  any metric beyond tolerance is blocked until explained or fixed.
- Reports are artifacts of record — the project's quality memory,
  complementing the commit log.

## Baselines and ablations (prove the math earns its keep)

- Floors: majority-class classifier, keyword rules, BM25-only answers.
  The engine must beat all of them.
- **Ablations per §6.1 mitigation**: remove negation features / citation
  features / lexicon / calibration one at a time and measure the delta.
  A mitigation that doesn't earn its delta gets revisited — requirements
  stay honest.

## Human evaluation

- Weekly sample of accepted answers + boundary-queue review.
- Rubric (binary checks): citation correct? answers the question?
  abstention appropriate? no legal advice? no personal data?
- Two raters on a subsample; disagreements refine the gold sets, not the
  rubric-of-the-week.

## Cadence and triggers

| Event | Action |
| ---------------------------------------- | ---------------------------- |
| model / corpus / lexicon / rules / thresholds change | full `make eval` |
| weekly | boundary review → gold growth |
| monthly | safety-set refresh (new adversarial phrasings) |
| tiny-LLM layer proposal | shadow eval + ablation with/without |

## Investment decisions are eval-triggered

LoRA/fine-tuning and tier upgrades are considered **only when a measured
bottleneck survives data/feature/threshold fixes** — see
[tiny-llm.md](tiny-llm.md) "Fine-tuning (LoRA)". Evals first, investment
second.