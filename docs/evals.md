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

- Split documents into **training / calibration / final holdout**, grouped
  by case. Masked and unmasked CNJ numbers in provenance identify the same
  case; normalized duplicate texts also stay together, transitively.
- Within each class, order groups by their latest publication date: newest
  20% of **dated groups** for final holdout, preceding 20% for calibration,
  remainder for training (at least one group in each). Undated statutes and
  authored seeds stay in training so they do not displace older cases.
  Entirely undated classes use the same proportions with stable group hashes
  to order groups. This is a per-class chronological split, not a single
  court-wide date cutoff.
- Require three independent dated groups per class when dates are present,
  or three undated groups otherwise; fail explicitly when there are fewer.
  Exclude unresolved labels and case-derived documents without an identifiable
  CNJ number. Conflicting labels within a group fail rather than being guessed.
- Fit vocabulary, IDF and weights on training only; temperature on calibration
  only; report metrics on final holdout only. Report all three sample counts.
- Query routing uses authored data only. Scenario variants stay together in
  calibration folds, with normalization learned inside each training fold.
  Development questions guide iteration; they are not an untouched test set.

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

## Retrieval by meaning (2026-11-25, retrieval-v1)

80 authored questions (40 dev, 40 hidden; not reviewed by a lawyer), on the
347 statute articles. The hidden half was committed before the ranking work
and not tuned against; its per-question results were never inspected.

| Ranking | Dev first | Dev top-5 | Hidden first | Hidden top-5 |
| --- | --- | --- | --- | --- |
| words (BM25) | 10 | 21 | 19 | 27 |
| **meaning (e5-small)** | **29** | **34** | **25** | **29** |
| bge-m3 (2+ GB, not adopted) | 32 | 38 | – | – |

Answers, with routing assumed correct (right / wrong / deferred): dev 27/6/7
by meaning against 10/26/4 by words; hidden 22/9/9 against 18/19/3. Reciprocal
rank fusion with BM25 was worse than meaning alone (dev first 17). Targets are
not met: recall@5 is 0.73 hidden against 0.90, and wrong answers remain.
Hidden has now been opened once: the next milestone needs a fresh set. Rerun
`agavus embed` after statutes change.

Quantized weights (int8, 118 MB) were measured and not adopted: dev first 27,
top-5 31, with no speed gain against the full model (470 MB: 29, 34).

## Question router by meaning (2026-11-25, routing-v2)

100 authored questions (50 dev, 50 hidden; not reviewed by a lawyer), committed
before the work. The 10-item gold set had hidden a safety problem: M recall
1.00 there was 5/20 and 2/20 here.

| Router | Set | Right | Wrong | Deferred | M recall | Unsafe |
| --- | --- | --- | --- | --- | --- | --- |
| words (v0.3.0) | dev | 8 | 1 | 41 | 5/20 | 1 |
| **meaning** | dev | **36** | 2 | **12** | **19/20** | **0** |
| words (v0.3.0) | hidden | 5 | 4 | 41 | 2/20 | 1 |
| **meaning** | hidden | **36** | 1 | **13** | **19/20** | **0** |

Unsafe = an M question accepted as F or rejected. Loss matrix and thresholds
are unchanged. The same author wrote seeds and questions (no near-duplicates).
Zero unsafe in 20 still allows a true rate up to about 14% (95%). Without
encoder weights every question is deferred (since v0.4.0). Space
releases now require the router gate (`publish/router_gate.py`).
More questions answered means more retrieval errors shown: retrieval dev with
real routing is 21 right, 5 wrong, 14 deferred (was 17/1/22).

## Evaluation repair and query expansion (2026-11-26)

The document pipeline previously fitted temperature and reported ECE on the
same holdout. It now separates calibration and final evaluation, preserves
case groups and dates, and excludes labels awaiting review. Legacy SQLite
reports retain their values with calibration count zero; they are not directly
comparable with the new reports.

The question router adds 64 authored synthetic queries in 32 paired scenarios
(330 queries total). New scenario variants stay together in calibration folds;
legacy seeds are grouped by normalized text. Each fold fits its own vector
centering and scale. The encoder, solver settings and decision costs are
unchanged: **M ceiling preserved**.

Measured locally using the pinned CPU ONNX encoder on the existing 50-question
**development** set:

| Metric | Previous | Revised |
| --- | --- | --- |
| Correct / wrong / deferred decisions | 36 / 2 / 12 | 42 / 1 / 7 |
| Accepted precision | 29/31 (93.5%) | 29/30 (96.7%) |
| Accepted coverage | 62% | 60% |
| Correct out-of-scope rejections | 7 | 13 |
| Argmax macro-F1 | 0.922 | 0.922 |
| ECE | 0.056 | 0.026 |
| M argmax recall | 19/20 | 19/20 |
| Unsafe M decisions | 0 | 0 |

The improvement is mainly more reliable out-of-scope rejection, not more
correct accepted answers. The 10-question routing-v1 regression gate also
passes: M recall and accepted precision 1.00, accepted coverage 0.50.
The larger set's 0.95 M recall still misses the 0.98 target. No hidden-set
predictions were rerun for this iteration; development results are not an
independent estimate of generalization. Jev checked the authored scenario
labels for routing consistency; this is not lawyer review.

Document retraining selected 651 resolved corpus labels plus 86 seeds.
An initial split allocated quotas across undated auxiliary data and dated
cases together, leaving family training without real cases: accuracy 0.333,
ECE 0.319. The mixed-source split now reserves calibration/evaluation quotas
from dated cases and retains undated auxiliary data in training. On 621
training / 58 calibration / 58 evaluation examples: accuracy 0.966,
macro-F1 0.648, ECE 0.031. Evaluation contains 34 family, 23 domestic-violence
and only **one** out-of-scope case. Weak organ labels are not expert gold.
The candidate also fails the query gold M-recall gate (0.75); it remains a
local candidate rather than replacing the document artifact. More independently
reviewed out-of-scope documents and a fresh query gold set remain necessary.
