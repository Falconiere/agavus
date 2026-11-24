# agavus — Design Document

Status: **design only, no implementation yet**.
Scope: CPU-only legal classification + retrieval/answer engine for Brazilian law,
restricted to **TJDFT** data and three domains.

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
| ---- | ------------------ | ------------------------------------------------ |
| F | Direito de família | divórcio, guarda, alimentos, união estável |
| M | Maria da Penha | Lei 11.340/2006, medidas protetivas, violência doméstica |
| P | Direito penal | crimes em geral, dosimetria, execução penal |
| O | Out of scope | everything else → reject |

Subtopics come from CNJ's **TPU** (Tabelas Processuais Unificadas) taxonomy,
filtered to these domains.

### Taxonomy nature: routing, not legal branches

F/M/P is a **routing taxonomy** (which procedure, urgency, and corpus), not a
taxonomy of legal branches:

- *Direito penal* and *direito criminal* are synonyms — one domain (P).
- **Maria da Penha (Lei 11.340/2006) is a hybrid special law**: penal facet
  (art. 24-A, CP art. 129 §13, CPP art. 313 III), civil/família facet
  (protective measures: removal from home, no-contact, temporary custody and
  alimentos), and procedural autonomy (protective measures independent of any
  criminal case — STJ case law, Lei 14.550/2023). TJDFT's specialized Varas de
  Violência Doméstica e Familiar handle both facets.

**Real overlap**: M↔F (guarda/alimentos inside protective measures) and M↔P
(lesão corporal, ameaça, feminicídio) genuinely co-occur. Consequences:

- Single-label routing resolves ambiguity through the **boundary region**:
  mixed M/F matters are deferred (clarify/escalate), never forced into one class.
- Evaluation watches the M↔F confusion cells explicitly (§8).
- If TJDFT data shows frequent co-occurrence, evolve to **multi-label**
  classification; the decision layer's expected-loss rule generalizes to
  label sets without structural change.

## 3. Data sources (TJDFT only)

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
   F/M/P/O via a curated code table (`taxonomy` domain).
2. **Weak labels from órgão julgador**: TJDFT specialized organs are strong
   domain signals:
   - Vara de Violência Doméstica e Familiar contra a Mulher → M
   - Vara de Família → F
   - Vara Criminal / Tribunal do Júri → P
   Rules are versioned and auditable; conflicts with TPU codes go to a review
   queue, never silently resolved.
3. **Manual gold set**: small curated set (queries + documents) for calibration
   and evaluation; the only human-labeled data.

## 5. Corpus schema (draft)

Frozen, typed records (implementation will live in `corpus/`):

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
| TPU code table → F/M/P mapping | `data/processed/` (cleaned, labeled) |
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

| Tier | Component | Math | Role |
| ---- | ------------------------- | ------------------------------------------------ | ------------------------- |
| 1 | TF-IDF + logistic regression | `tfidf(t,d) = count(t,d)·log(N/df(t))`; softmax, cross-entropy | domain classifier (probabilities) |
| 1 | Linear SVM / Naive Bayes | max-margin hinge / multinomial NB | baselines to beat |
| 2 | BM25 + cosine + citation features | saturating TF, length normalization; regex citations | retrieval |
| 3 | NMF/LDA, HDBSCAN, PageRank | matrix factorization; density clustering; citation graph | subtopic discovery, authority ranking |
| 4 | Small CPU embeddings (MiniLM-class) | cosine over dense vectors | semantic retrieval upgrade (later) |
| 5 | Quantized 0.5–1.5B via llama.cpp | CPU inference | optional fluency over retrieved facts (later) |

- Word + char n-grams (Portuguese morphology/typos); binary citation features.
- **Probability calibration** (Platt/isotonic on held-out set) is mandatory:
  the decision layer consumes calibrated probabilities only.

## 7. Decision layer: three-way decisions (Yao / DTRS)

Actions: **accept** (classify/answer), **reject** (out of scope), **defer**
(clarifying question, second retrieval pass, or human review queue).

Loss matrix (units: 1 = cost of one deferral round; placeholders to calibrate):

| Action ↓ / True state → | F | M | P | O |
| ----------------------- | --- | --- | --- | --- |
| Accept as F | 0 | 10 | 4 | 8 |
| Accept as M | 6 | 0 | 3 | 8 |
| Accept as P | 4 | 3 | 0 | 8 |
| Reject (out of scope) | 4 | 8 | 4 | 0 |
| Defer | 1 | 1 | 1 | 1 |

Rationale:

- M→F = 10: worst error — a domestic-violence matter treated as ordinary
  family dispute can delay protective measures (safety-critical).
- Reject|M = 8: never silently drop a possible Maria da Penha case.
- F→M = 6: false violence framing is serious but less urgent than the reverse.
- Accept|O = 8: answering out of scope = hallucinated authority.

Decision rule: pick the action minimizing expected loss
`Σ λ(action|state) · P(state|x)` over calibrated probabilities.

Interpretable thresholds (binary reduction per domain,
`α = (λ_PN − λ_BN)/((λ_PN − λ_BN) + (λ_BP − λ_PP))`,
`β = (λ_BN − λ_NN)/((λ_BN − λ_NN) + (λ_NP − λ_BP))`):

| Domain | α (accept ≥) | β (reject ≤) | Behavior |
| --------------- | ------------ | ------------ | -------------------------------------- |
| Família | ~0.86 | ~0.25 | strictest acceptance (M→F catastrophic) |
| Maria da Penha | ~0.80 | ~0.12 | hardest to reject; weak signals defer |
| Penal | ~0.80 | ~0.25 | balanced |

Applied at two gates: routing (classify) and answering (answer/abstain/escalate).

## 8. Evaluation plan

- **Classifier**: macro-F1 per domain on held-out TJDFT data; confusion matrix
  with special attention to M↔F errors.
- **Decision layer**: accuracy-vs-coverage curves while sweeping α/β; target
  e.g. ≥ 0.95 precision on accepted answers; track deferral rate and boundary
  queue composition; review sampled boundary cases weekly.
- **Retrieval**: recall@k on the gold question set; citation correctness
  (every excerpt must carry a verifiable source reference).
- **Safety**: zero tolerance for party-name leakage (automated scan);
  out-of-scope rejection rate on an O-set.

## 9. Roadmap (maps to repo domain layout)

1. `corpus/` — DataJud + DJEN ingestion for TJDFT, anonymization, schema;
   gitignored `data/` + `AGAVUS_DATA_DIR` setting + staged-data guard hook.
2. `taxonomy/` — TPU code table → F/M/P/O; weak-label rules; gold set.
3. `classify/` — TF-IDF + calibrated logreg; baselines; evaluation.
4. `decide/` — loss matrix, expected-loss rule, three-way gates.
5. `retrieve/` — BM25 + citation features over statutes/decisions.
6. `answer/` — extractive, cited answers; abstain/escalate paths.
7. Later: tiers 4–5, subtopic discovery, other tribunals.

All modules follow repo rules: strict types, co-located tests, 2 spaces,
size limits, zero duplication, `make check` green.