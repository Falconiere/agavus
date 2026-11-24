# Evals — guaranteeing and benchmarking quality

Status: **harness live** (`agavus eval`, gold set v1, release gate in
`publish/export.gate`). Principle: **no change ships without measured evidence.**
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

- `agavus eval` runs the gold dimensions → console report; `--json` for the
  typed report; `agavus publish` gates releases on it (metrics, set
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
| model / corpus / lexicon / rules / thresholds change | full `agavus eval` |
| weekly | boundary review → gold growth |
| monthly | safety-set refresh (new adversarial phrasings) |
| tiny-LLM layer proposal | shadow eval + ablation with/without |

## First real training runs (2026-11-24)

Recorded as the project's quality memory (commits-as-memory principle):

1. **Statutes-only model** (61 articles): holdout acc 1.00 — but colloquial
   queries misrouted ("divórcio e guarda" → M, because Lei 11.340 also
   mentions guarda/alimentos in protective measures). Lesson: **domain
   shift** — statute language ≠ query language; perfect in-domain holdout
   does not imply query competence.
2. **+ authored seed queries** (`classify/seed.py`, 32 queries incl. OOD,
   firewalled from gold): "divórcio e guarda" → F 0.66 ✓; "descumpriu
   protetiva" → M 0.90 ✓; "tráfico" → O 0.41 (weak); vague custody query →
   F 0.49 / M 0.41 (ambiguous). Holdout acc 0.79, macro-F1 0.54, ECE 0.069.

Interpretation: low-confidence and ambiguous cases are **exactly what the
three-way decide gate must defer** — the probabilities are now meaningful
inputs for it. Next: `decide/` + gold-set evaluation (`agavus eval`).

## First decision-layer eval (2026-11-24, routing-v1)

`agavus eval` with the three-way gate active:

- acc 0.90 · macro-F1 0.85 · **M-recall 1.00** (safety target ≥ 0.98 ✓)
- decisions: 1 accepted (1 correct) · 9 deferred · 0 rejected
- **precision-on-accepted 1.00 · coverage 0.10**

Interpretation: the gate is doing its job — with a statute+seed-trained
model, almost everything is honestly deferred instead of guessed. The
accuracy/coverage dial is visible: coverage grows with training data
(DJEN decisions) and features, **never by weakening safety thresholds**.

## Founding query milestone (2026-11-24)

`agavus -p "Qual é o artigo da lei de alienação parental?"` → **accept**,
verbatim Lei 12.318/2010 art. 1 excerpt + context refs + disclaimer, exit 0.
Reached by adding provision-question seeds (data, not threshold tuning):
before them the query sat at F 0.888 vs α_F 0.889 and was honestly deferred
— the gate working as designed. Eval after: M-recall 1.00,
precision-on-accepted 1.00, coverage 0.20.

## Corpus encoding findings (2026-11-24)

Planalto serves mixed encodings: cp1252, UTF-8, and **UTF-16** (Word-generated
compilado pages, sometimes with a truncated trailing byte). `decode_html`
detects BOMs and tolerates truncation. Firecrawl-rendered copies of these
pages lose bytes irreversibly (U+FFFD in tag regions) — **always import raw
payloads, never rendered copies**. Article segmentation captures letter
suffixes (art. 24-A) and trims bodies at chapter headings.

## Growth run (2026-11-24): corpus 89 → 355, coverage 0.20 → 0.60

Levers: CC família book (arts. 1.511–1.783, 266 articles), Lei 13.058/2014
(guarda), Lei 14.994/2024 (feminicídio package), +15 seeds. DJEN remains
blocked from the dev sandbox (full-text decisions pending).

**Safety regression caught and fixed by the eval gate**: the F-heavy corpus
drove M-recall 1.00 → 0.25 (class imbalance). The decision gate kept
precision 1.00 (M queries deferred, never misrouted), and the fix was
**class balancing in the solver** (`TrainConfig.balance_classes`) — a
modeling fix, never threshold weakening.

| Metric | Before growth | Imbalanced | Balanced (final) |
| ---------------------- | ------------- | ---------- | ---------------- |
| gold accuracy | 0.80 | 0.50 | **1.00** |
| M-recall | 1.00 | 0.25 ✗ | **1.00** ✓ |
| precision-on-accepted | 1.00 | 1.00 | **1.00** ✓ |
| coverage | 0.20 | 0.30 | **0.60** |
| holdout ECE | 0.156 | 0.067 | **0.048** |

OOD gold items now correctly **rejected** (2), ambiguous honestly deferred
(2), 6 accepted all correct. Founding query answered with citation.

## Investment decisions are eval-triggered

LoRA/fine-tuning and tier upgrades are considered **only when a measured
bottleneck survives data/feature/threshold fixes** — see
[tiny-llm.md](tiny-llm.md) "Fine-tuning (LoRA)". Evals first, investment
second.