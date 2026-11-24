# agavus — Design Document

Status: **MVP implemented and published** (corpus → taxonomy → classify →
decide → retrieve → reason → answer; HF model repo + Space live).
Scope: CPU-only legal classification + retrieval/answer engine for Brazilian law,
restricted to **TJDFT** data and two domains.

## 1. Goals

1. **Document classifier**: route legal text to the correct domain/subtopic.
2. **Search/answer tool**: answer questions with *actual cited excerpts*
   (statutes + decisions), never generated "opinions".

Constraints:

- No GPU. Training must run on CPU in minutes; inference cheap.
- Pure-mathematical core (tiers 1–3 below); small CPU inference / API models
  (tiers 4–5) allowed later as optional layers, never as source of truth.
- No legal advice: answers are cited excerpts + abstention/escalation paths.
- LGPD: anonymize parties; never expose personal data from case records.

## 2. Domain scope

| Code | Domain | Examples |
| ---- | ------------------------------------ | ------------------------------------------------ |
| F | Direito de família | divórcio, guarda, alimentos, união estável |
| M | Maria da Penha / violência doméstica | medidas protetivas **and** DV criminal prosecutions: lesão corporal, ameaça, feminicídio, art. 24-A |
| O | Out of scope | everything else, **including general criminal law** → reject |

Subtopics come from CNJ's **TPU** (Tabelas Processuais Unificadas) taxonomy,
filtered to these domains. Inside M, TPU subtopics preserve the distinction
between protective measures (civil/autonomous) and criminal prosecutions —
routing is binary, detail is not lost.

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
  mixed M/F matters are deferred (clarify/escalate), never forced into one class.
- Evaluation watches the M↔F confusion cells explicitly (§8).
- If TJDFT data shows frequent co-occurrence, evolve to **multi-label**
  classification; the decision layer's expected-loss rule generalizes to
  label sets without structural change.

## 3. Data sources (TJDFT only)

Deep-dive documentation: **[docs/corpus/](corpus/README.md)** (range, sources,
schema, ingestion, labeling, privacy, storage).

**Time window**: case metadata and decisions from **2020-01-01 → current
date at ingestion** (rolling end, incremental runs; `AGAVUS_DATA_START`
configurable). Rationale: captures Lei 14.550/2023 and Lei 14.994/2024
reforms, born-digital records, manageable volume. Statutes are the exception:
current consolidated text, not windowed. Train/held-out split by case,
preferably by time.

| Source | Content | Use |
| ------------------------- | -------------------------------------------- | ------------------------------ |
| DataJud API (CNJ) | case metadata, filtered `tribunal = TJDFT` | labels (TPU codes), features |
| DJEN | full-text decisions, filtered to TJDFT | training text, retrieval corpus |
| TJDFT jurisprudence portal | acórdãos/decisões search | retrieval corpus supplement |
| Planalto / LexML | statutes (CP, CC Livro IV, Lei 11.340/2006) | retrieval corpus, citation features |

Caveats:

- **Segredo de justiça**: family and Maria da Penha full texts are often
  sealed. Expect metadata-heavy data for F/M, full-text-heavy for P.
- Single court = consistent vocabulary; generalization to other tribunals is
  an explicit later phase (domain shift), not an MVP concern.

## 4. Labeling strategy

1. **Free labels**: DataJud records carry TPU classe/assunto codes → map to
   F/M/O via a curated code table (`taxonomy` domain).
2. **Weak labels from órgão julgador**: TJDFT specialized organs are strong
   domain signals:
   - Vara de Violência Doméstica e Familiar contra a Mulher → M
   - Vara de Família → F
   - Vara Criminal → O (general criminal is out of scope)
   - Tribunal do Júri → O, unless TPU subjects confirm feminicídio/DV
     context → M
   Rules are versioned and auditable; conflicts with TPU codes go to a review
   queue, never silently resolved.
3. **Manual gold set**: small curated set (queries + documents) for calibration
   and evaluation; the only human-labeled data.

## 5. Corpus schema (implemented in `corpus/records.py`)

Frozen, typed records:

- `CaseRecord`: id, tribunal, orgao_julgador, classe_tpu, assuntos_tpu[],
  datas, movimentos, domain_label, label_source (tpu | organ | manual | weak).
- `TextDocument`: id, source (djen | portal | statute), origin_ref, text,
  citations[] (regex-extracted `art. X, Lei Y`), domain_label, label_source.
- All party names stripped at ingestion (LGPD); ingestion is idempotent.

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
- Ingestion is idempotent and resumable: the corpus is a rebuildable cache,
  not an asset — losing it costs a re-fetch, nothing more.
- Hard guard: a pre-commit hook fails if anything under `data/` is staged,
  on top of `.gitignore`.
- Later, if sharing/versioning is needed: object storage (S3/R2) or a private
  dataset host, referenced by config — never by git. DVC-style snapshot
  versioning only if reproducibility demands it (skip for MVP).

## 6. Model stack

Formula-by-formula explanations with worked examples: **[docs/math/](math/README.md)**.
End-to-end data-to-prediction flow: **[math/classification-pipeline.md](math/classification-pipeline.md)**.

| Tier | Component | Math | Role |
| ---- | ------------------------- | ------------------------------------------------ | ------------------------- |
| 1 | TF-IDF + logistic regression | `tfidf(t,d) = count(t,d)·log(N/df(t))`; softmax, cross-entropy | domain classifier (probabilities) |
| 1 | Linear SVM / Naive Bayes | max-margin hinge / multinomial NB | baselines to beat |
| 2 | BM25 + cosine + citation features | saturating TF, length normalization; regex citations | retrieval |
| 3 | NMF/LDA, HDBSCAN, PageRank | matrix factorization; density clustering; citation graph | subtopic discovery, authority ranking |
| 4 | Small CPU embeddings (MiniLM-class) | cosine over dense vectors | semantic retrieval upgrade (later) |
| 5 | Quantized 0.5–4B via llama.cpp | CPU inference | optional bounded proposer behind a verifier (later, [tiny-llm.md](tiny-llm.md)) |

- Word + char n-grams (Portuguese morphology/typos); binary citation features.
- **Probability calibration** (Platt/isotonic on held-out set) is mandatory:
  the decision layer consumes calibrated probabilities only.

### 6.1 Known limitations → required mitigations

These are **implementation requirements**, not optional improvements. Each
limitation of the math core has a designated fix; mitigations live in
preprocessing/features — deterministic, inspectable, CPU-only. Auditability
beats opaque accuracy in legal text.

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

## 7. Decision layer: three-way decisions (Yao / DTRS)

Actions: **accept** (classify/answer), **reject** (out of scope), **defer**
(clarifying question, second retrieval pass, or human review queue).

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

Applied at two gates: routing (classify) and answering (answer/abstain/escalate).

## 8. Evaluation plan

Full plan, gold sets, targets, and regression policy:
**[docs/evals.md](evals.md)**. Summary:

- **Classifier**: macro-F1 per domain on held-out TJDFT data; confusion matrix
  with special attention to M↔F errors; expected calibration error (ECE)
  reported alongside F1.
- **Decision layer**: accuracy-vs-coverage curves while sweeping α/β; target
  e.g. ≥ 0.95 precision on accepted answers; track deferral rate and boundary
  queue composition; review sampled boundary cases weekly. Reviewed boundary
  cases flow back into the gold set (active learning: most-uncertain first).
- **Retrieval**: recall@k on the gold question set; citation correctness
  (every excerpt must carry a verifiable source reference).
- **Safety**: zero tolerance for party-name leakage (automated scan);
  out-of-scope rejection rate on an O-set.

## 9. Roadmap (maps to repo domain layout)

Extension to new areas (trabalhista, bancário, …) is a first-class design
concern: **[docs/extensibility.md](extensibility.md)** — areas are data packs
(AreaSpec registry), the engine is invariant. Seams honored from day one:
labels via registry, loss matrix as composed data, namespaced per-area
storage, area-parameterized pipelines.

1. `corpus/` — ✅ DataJud + statute ingestion (offline import), anonymization,
   schema, SQLite store; gitignored `data/` + `AGAVUS_DATA_DIR` + guard hook.
   ⬜ DJEN full-text decisions (network-blocked in dev sandbox).
2. `taxonomy/` — ✅ area registry, organ weak-label rules, statute map,
   lexicon, gold-set seed; ⬜ TPU code table populated from real DataJud codes.
3. `classify/` — ✅ TF-IDF (word-internal char n-grams) + balanced, calibrated
   logreg; negation/citation/verb features; seeds; ⬜ feature crosses, baselines.
4. `decide/` — ✅ loss matrix, expected-loss rule, three-way gates, safety
   ceiling; ⬜ boundary queue feeding the gold set automatically.
5. `retrieve/` — ✅ FTS5 BM25 + law-name column + citation boost;
   ⬜ hybrid dense upgrade (tier 4).
6. `reason/` — ✅ facts + rule pack + verified chains (R-24A, R-22);
   ⬜ citation graph, temporal reasoning, more rules.
7. `answer/` — ✅ extractive cited answers, degradation ladder, topic-targeted
   clarification; ⬜ relevance scoring, advice lint.
8. ⬜ Later: tiers 4–5 ([tiny-llm.md](tiny-llm.md)), subtopic discovery,
   other tribunals.

Cross-cutting from step 3 onward: `evals/` harness + gold sets
([evals.md](evals.md)) — no change ships without measured evidence;
LoRA/tier upgrades are eval-triggered investments, never advance capabilities.
Publishing to Hugging Face (model repo + CPU Space, eval-gated, LGPD-bound):
**[docs/publishing.md](publishing.md)**.

All modules follow repo rules: strict types, tests in per-domain `tests/`
folders, 2 spaces, size limits, zero duplication, `make check` green.

## 10. CLI usage contract

The product surface is defined in **[docs/cli.md](cli.md)**:
`agavus -p "pergunta"` (ask → classify → decide → retrieve → cited excerpt),
`agavus classify` for documents, `--explain` auditability trace, `--json`
typed output, scriptable exit codes (0 accept / 10 defer / 20 reject).
pt-BR user I/O; fixed no-legal-advice disclaimer; sources never empty.
Delivery contract (pre-delivery checklist, degradation ladder, response-level
evals): **[docs/response-quality.md](response-quality.md)**.

## 11. Reasoning and the tiny-LLM layer

- **Reasoning**: verified chains — symbolic syllogism, requirements
  checklists, citation graph, temporal logic. Every conclusion carries its
  cited premises; broken chains defer. **[docs/reasoning.md](reasoning.md)**.
- **Tiny CPU LLM** (0.5–4B quantized, llama.cpp): optional bounded proposer
  behind a verifier — transformer, not oracle. Disabled by default,
  shadow-gated, cached; the classical core is complete without it.
  **[docs/tiny-llm.md](tiny-llm.md)**.