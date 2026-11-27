# agavus — Design Document

The engine supports document classification, question routing, retrieval and cited answers.
The public demo and model repository are linked in the [README](../README.md).
Scope: CPU-only legal classification + retrieval/answer engine for Brazilian law,
restricted to **TJDFT** data and two domains.

## 1. Goals

1. Route documents and questions to family law, domestic violence or outside scope.
2. Return cited excerpts or a rule conclusion with its statutory premise. Defer when
   the evidence is insufficient.

Production inference runs on CPU. The private learned-classifier comparison can also
train on local MPS. Research artifacts remain isolated from production. Protect
case-derived data and follow the [corpus privacy policy](corpus/privacy.md).

## 2. Domain scope

| Code | Domain | Examples |
| ---- | ------------------------------------ | ------------------------------------------------ |
| F | Direito de família | divórcio, guarda, alimentos, união estável |
| M | Maria da Penha / violência doméstica | medidas protetivas **and** DV criminal prosecutions: lesão corporal, ameaça, feminicídio, art. 24-A |
| O | Out of scope | everything else, **including general criminal law** → reject |

The production output is a single area decision. Case metadata can retain subject codes.
The engine does not produce a subtopic prediction.

### Taxonomy nature: routing, not legal branches

F/M is a **routing taxonomy** (which procedure, urgency, and corpus), not a
taxonomy of legal branches:

- **Maria da Penha (Lei 11.340/2006) is a hybrid special law**: penal facet
  (art. 24-A, CP art. 129 §13, CPP art. 313 III), civil/família facet
  (protective measures: removal from home, no-contact, temporary custody and
  alimentos), and procedural autonomy (protective measures independent of any
  criminal case — STJ case law, Lei 14.550/2023). TJDFT's specialized Varas de
  Violência Doméstica e Familiar handle both facets.
- **M deliberately absorbs the whole DV picture** — both facets route to M.
  General criminal law (crimes outside the DV context) is out of scope (O).

**Real overlap**: M↔F (guarda/alimentos inside protective measures) genuinely
co-occurs. Consequences:

- Single-label routing resolves ambiguity through the **boundary region**:
  mixed M/F matters can defer when the expected loss of an area decision is high.
- Evaluation watches the M↔F confusion cells explicitly (§8).
- If TJDFT data shows frequent co-occurrence, evolve to **multi-label**
  classification; the decision layer's expected-loss rule generalizes to
  label sets without structural change.

## 3. Production data sources (TJDFT only)

The private decision-reading pilot may compare public family and domestic-violence
decisions from TJSP and TJRJ with the TJDFT sample. Keep each court's sources, dates,
deciding bodies and findings identifiable in the ignored experiment directory.
Production ingestion, training and published retrieval retain their TJDFT scope.

Deep-dive documentation: **[docs/corpus/](corpus/README.md)** (range, sources,
schema, ingestion, labeling, privacy, storage).

**Time window**: no start date. Ingestion starts at the current date and
walks backward until the source has no more rows for the subject (F and M,
TJDFT). `AGAVUS_DATA_START` still loads so older env files do not break;
fetches ignore it. Statutes stay the current consolidated text. Train/held-out
split by case, preferably by time, on whatever span was stored.

The three implemented source adapters and their coverage limits are documented in
[corpus sources](corpus/sources.md). Statute articles and published decisions can
support cited answers. Neither a court publication nor a routing probability
establishes the likelihood that an individual protective order will be revoked.

## 4. Labeling strategy

Case labeling tries mapped TPU subjects, then the issuing court. The shipped TPU
table has no verified entries yet. Conflicting signals require review. Text labeling
uses the curated statute map or an organ named in the reference. Training reads
resolved labels and excludes those flagged for review. [Detailed policy](corpus/labeling.md)

## 5. Corpus schema (implemented in `corpus/records.py`)

CaseRecord holds metadata, dates, movements and provenance. TextDocument holds a
source type, citable reference, text and extracted citations. Both allow an unresolved
area and record the source of their label. [Fields and types](corpus/schema.md)

### Corpus storage policy: never in git

The repo is **public**; real case data must never be committed, even
anonymized (re-identification risk). Split:

| In the repo | Outside git |
| ------------------------------------------ | -------------------------------------- |
| ingestion code + typed schema | `data/raw/` (as fetched) |
| TPU code table → F/M/O mapping | `data/processed/` (cleaned, labeled) |
| tiny **synthetic** test fixtures | anything derived from real cases |

- Corpus lives in `data/` (gitignored), location configurable via
  `AGAVUS_DATA_DIR` in the isolated settings module (default: `data/`).
- Deterministic IDs deduplicate repeated source/reference keys. Preserve private raw
  evidence and manifests where reproducibility matters.
- The staged-data guard rejects files under `data/`.

## 6. Model stack

Formula-by-formula explanations with worked examples: **[docs/math/](math/README.md)**.
End-to-end data-to-prediction flow: **[math/classification-pipeline.md](math/classification-pipeline.md)**.

| Entry point | Representation and decision |
| --- | --- |
| Document classification | Word and character TF-IDF, legal features and class-balanced logistic regression with temperature scaling |
| Question routing | Pinned multilingual E5 encoder on CPU, linear classifier with temperature fitted on out-of-fold logits |
| Search | SQLite FTS5 BM25 with citation boosts |
| Accepted question | Area-specific rule and cited premise, or statute search prioritizing explicit citations and ranking by meaning. Lexical statute search is available when the index is missing. |

The CPU embedding layer (tier 4 in the design) serves question routing and statute ranking.
The question entry point requires the encoder for routing and defers when it is unavailable.
The cost matrix consumes calibrated area probabilities. [Math and training splits](math/classification-pipeline.md)

### 6.1 Known limitations → required mitigations

These requirements govern work on the production core. The model table above describes
what is implemented. The mitigation table also records design obligations, including
review feedback, comparison of retrieval methods and calibration of costs. Their
inclusion here does not indicate a deployed queue or a live fusion algorithm.

| # | Limitation | Required mitigation | Module |
| - | ---------------------------------------- | ------------------------------------------------------------------ | ------------------ |
| 1 | Negation/word order lost ("nega a protetiva" ≈ "concede a protetiva") | Negation-aware features: closed lexicon of legal decision verbs (deferir/indeferir, conceder/negar, prover/desprover, revogar, homologar); NegEx-style scope marking so `nega_protetiva` ≠ `protetiva`; bigrams/trigrams | `classify/features` |
| 2 | Lexical gap (alimentos vs pensão alimentícia; abbreviations LMP, VP) | Curated synonym/abbreviation → canonical-term lexicon applied at preprocessing and query expansion; LSA/NMF latent topics (tier 3); hybrid BM25 + dense with reciprocal rank fusion (tier 4) | `taxonomy` (lexicon), `retrieve` |
| 3 | Linear model misses feature interactions | Explicit feature crosses (citation × domain, negation × term, bigrams); model stays linear for auditability | `classify/features` |
| 4 | Calibration needs data we don't have yet | Cross-validated Platt on the small gold set; **defer-queue reviews feed the gold set** (human-in-the-loop); active learning — label most-uncertain examples first | `classify`, `decide` |
| 5 | Loss matrix values are placeholders | Calibrate from held-out α/β sweeps and boundary-queue review outcomes; version every change | `decide/losses` |

Fallback for #1 only if rules prove insufficient: CPU dependency parsing
(spaCy pt) for negation scope — heavier, still no GPU.

Architectural invariant: even perfect NLU would not remove the need for
**citations + defer**. Correctness comes from quoting real sources and
abstaining, not from deeper understanding.

#### Private learned-classifier alternative

`classify/learned/` may learn negation, wording and interactions from supervised
examples instead of the lexicons and feature crosses above. This exception is
limited to the private experiment. The §6.1 requirements continue to govern production.
Raw pretrained-tokenizer input feeds a trainable, pinned multilingual E5-small
encoder with masked mean pooling and separate linear area and statement-relation
heads. Relation labels (support, contradiction, unknown) are internal supervision,
never routing areas or generated legal explanations. No runtime cue rules apply
in this path.

Compare TF-IDF, frozen E5, area-only fine-tuning and joint fine-tuning on one
frozen 60/15/10/15 training/validation/calibration/final split. Cases, duplicate
texts and synthetic template families stay together. Undated statutes and existing
query seeds are training-only auxiliaries. Fit temperatures only on calibration;
select checkpoints only on validation; report locked final results without tuning.
Exclude unresolved labels and residual personal data. Weak labels retain their
provenance. Additional negative cases are snapshot-only organ-derived labels.

Use 480-token document windows with 64-token overlap and a prefix in every
window; average window logits and normalize each document's training contribution.
Reject oversized statement pairs without truncation. AdamW uses encoder/head
rates 2e-5/1e-3, equal task weighting and training-only class balancing.

The optional `learned` extra stays outside application imports. All artifacts,
provenance, split membership and structured SQLite state stay under ignored
`data/experiments/learned/`; nothing is uploaded or replaces live artifacts.
The two-hour local computation ceiling includes pilot, encoding, training,
calibration and evaluation. Reserve evaluation time first, then choose a common
epoch count (at most three) for seeds 0, 1 and 2. Incomplete comparisons have no
winner. A candidate must beat both baselines, preserve M recall, have zero observed
unsafe M decisions, ECE ≤ 0.05 and accepted precision ≥ 0.95; report the separate
M recall ≥ 0.98 target. Synthetic and weak-label results cannot establish safety.
M ceiling preserved; citations, deferral and the existing loss matrix remain mandatory.
Execution, formulas and artifact layout: [private learned classifier](math/learned-classifier.md).

## 7. Decision layer: three-way decisions (Yao / DTRS)

Actions: **accept** (classify/answer), **reject** (out of scope), **defer**
(clarification or insufficient evidence).

Loss matrix (units: 1 = cost of one deferral round; placeholders to calibrate):

| Action ↓ / True state → | F | M | O |
| ----------------------- | --- | --- | --- |
| Accept as F | 0 | 10 | 8 |
| Accept as M | 6 | 0 | 8 |
| Reject (out of scope) | 4 | 8 | 0 |
| Defer | 1 | 1 | 1 |

Rationale:

- M→F = 10: worst error — a domestic-violence matter treated as ordinary
  family dispute can delay protective measures (safety-critical).
- Reject|M = 8: never silently drop a possible Maria da Penha case.
- F→M = 6: false violence framing is serious but less urgent than the reverse.
- Accept|O = 8: answering out of scope (incl. general criminal) =
  hallucinated authority.

Decision rule: pick the action minimizing expected loss
`Σ λ(action|state) · P(state|x)` over calibrated probabilities.

Interpretable thresholds (binary reduction per domain,
`α = (λ_PN − λ_BN)/((λ_PN − λ_BN) + (λ_BP − λ_PP))`,
`β = (λ_BN − λ_NN)/((λ_BN − λ_NN) + (λ_NP − λ_BP))`):

| Domain | α (accept ≥) | β (reject ≤) | Behavior |
| --------------- | ------------ | ------------ | -------------------------------------- |
| Família | ~0.89 | ~0.25 | strictest acceptance (M→F catastrophic) |
| Maria da Penha | ~0.86 | ~0.12 | hardest to reject; weak signals defer |

Both routing paths minimize the full multiclass expected loss. The binary thresholds
above illustrate the costs. Accepted questions also undergo evidence and delivery checks.

## 8. Evaluation

Document training reports final-holdout accuracy, macro-F1 and calibration error.
Release checks use authored routing examples. Space releases also test the question
router on development scenarios. Retrieval has a separate article-level evaluation.
Record coverage alongside precision and track M recall and unsafe routing explicitly.
[Evaluation targets, measured results and gates](evals.md)

## 9. Extension and research boundaries

Keep labels in the area registry and validate the composed loss matrix when extending it.
New courts or domains require an explicit scope decision. Private cross-court
decision reading retains tribunal, instance, issuing body and source on each record.
A forecast of revocation would require consecutive requests, information available
at filing and linked later outcomes, with transfers and unknown dispositions retained.
The public engine supplies area classification and cited retrieval.
[Area registry and extension requirements](extensibility.md)

## 10. CLI

Use `agavus classify TEXT` for a document, `agavus search QUERY` for lexical search
and `agavus -p QUESTION` for cited answers. The CLI documents each output and exit code.
[CLI](cli.md) · [Answer contract](response-quality.md) · [Publishing](publishing.md)

## 11. Reasoning and the learning lab

The engine matches grounded facts to three area-scoped rules and retrieves the cited
provision before delivering a rule conclusion. Language-model training takes place in
an isolated lab. The local Python server in `src/agavus/server/` serves the static chat
page and streams CPU continuations. Conversations belong to the browser. Production answers continue to
use the cited retrieval and deferral pipeline.
[Reasoning](reasoning.md) · [Lab isolation and data rules](llm-lab.md)
