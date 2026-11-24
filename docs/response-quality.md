# Response quality — the delivery contract

Status: **implemented**. Live by construction and pre-delivery verifier
(`answer/compose.py`): cited extractive answers, sources-never-empty,
degradation ladder, disclaimer, topic-targeted clarifications, content-token
and citation relevance check, privacy scan, and advice-language lint.
The front-end guarantee: **a response is delivered only
after passing automated checks; otherwise the engine degrades gracefully.**
Complements [cli.md](cli.md) (surface) and [evals.md](evals.md) (measurement).
Module: `answer/`.

## Anatomy of a high-quality response

```
1. Direct answer first   → the provision/excerpt that answers the question
2. Verbatim excerpt      → never paraphrased law
3. Precise citation      → lei, art., parágrafo, inciso + text version/date
4. Context (optional)    → related provisions / interpreting precedent, cited
5. Transparency          → decision + confidence (--explain: full chain)
6. Disclaimer            → fixed, always present
```

Example (the founding query):

```
Lei 12.318/2010, art. 2º — é o artigo que define alienação parental.

  "Considera-se ato de alienação parental a interferência na formação
   psicológica da criança ou do adolescente…"  (texto compilado, vigente)

Contexto: art. 4º (medidas cabíveis), art. 6º (exemplos de condutas).
Isto não é aconselhamento jurídico; consulte um advogado.
```

## Pre-delivery checklist (all automated, all must pass)

| # | Check | Failure action |
| - | -------------------------------------------- | ------------------------ |
| 1 | Decision gate: accept (three-way rule) | defer / reject |
| 2 | Citation validity: provision exists in corpus; excerpt matches source **verbatim** | drop to raw excerpt |
| 3 | Groundedness: every claim span-supported by cited text | drop to raw excerpt |
| 4 | Relevance: excerpt answers the *actual* question (intent↔excerpt match, threshold calibrated on answer-set) | clarify |
| 5 | Completeness: each part of the question addressed, or explicitly flagged as unanswered | clarify the missing part |
| 6 | Freshness: statute text is the current consolidated version; temporal flag when facts predate an amendment | annotate version |
| 7 | Privacy scan: zero personal data | block + log |
| 8 | Advice-language lint: no "você deve processar", "recomendo", outcome promises | block + rewrite to neutral |
| 9 | Disclaimer present | attach |

Checks 2–3 are exact (string/span verification) — possible because answers
are extractive. Checks 4–5 are heuristics **calibrated and gated by evals**,
never trusted blindly.

## The degradation ladder (never skip rungs silently)

```
verified composed answer → raw excerpt + citation → clarifying question → honest refusal
```

Every response states what it is. "Não encontrei base segura para responder"
is a **valid, high-quality outcome** — a confident wrong answer is the only
failure mode we treat as unacceptable.

## Conversation quality

- **Clarification loop**: defer produces a specific question with options
  ("guarda/visitas ou pensão?"); the answer re-enters the pipeline with
  context preserved.
- **Feedback capture**: every response offers "respondeu à sua pergunta?"
  — negative feedback routes to the boundary queue and feeds gold sets
  (the evals growth loop).
- **Follow-ups** reuse the established area/context; scope changes re-run
  the decision gate (never inherited trust).

## Latency and transparency

- Classical path is fast (CPU, milliseconds–seconds); show stage progress
  for longer flows: *classificando → buscando → verificando*.
- Tiny-LLM layer (later) streams under a p95 budget; if it exceeds budget,
  the classical answer ships — the LLM never holds the response hostage.
- `--explain` exposes the whole chain: probabilities, losses, features,
  retrieval scores, verification results. Quality you can audit.

## Response-level evaluation (beyond component metrics)

Added to the answer-set rubric (see [evals.md](evals.md)):

| Dimension | Check | Target |
| --------------- | ------------------------------------------ | ---------- |
| Relevance | answers the question asked | ≥ 0.95 |
| Completeness | all question parts addressed or flagged | ≥ 0.95 |
| Citation precision | right provision, verbatim, current | **1.00 (hard)** |
| Readability | pt-BR clarity, direct-answer-first | human rubric |
| Honesty | uncertainty visible, no hidden gaps | **1.00 (hard)** |

User feedback is a live metric: negative-feedback rate tracked per area,
reviewed weekly, traced to root cause (retrieval? relevance? scope?).

## Forbidden anti-patterns

1. Generated prose about law without span support.
2. Stale statute text presented as current.
3. Partial citations ("Lei Maria da Penha" without article).
4. Answering an *adjacent* question instead of the asked one.
5. Advice language or outcome promises.
6. Silent uncertainty — confidence must be visible or the response defers.