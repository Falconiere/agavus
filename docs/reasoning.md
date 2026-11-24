# Reasoning — verified chains, not generated opinions

Status: **Level 1 MVP implemented** (`reason/facts.py`, `reason/rules.py`,
`reason/chain.py`, wired into `answer/compose.py`). Levels 2–3 remain design.
Principle: **reasoning is only as good as its cited premises** — every hop
verifiable, broken chains defer. Companion: [tiny-llm.md](tiny-llm.md).

## Implemented (Level 1 MVP)

- **Extractive facts**: conservative patterns + negation guard; every fact
  carries its verbatim query span (`medida_vigente`, `descumprimento`,
  `violencia_domestica`, `contexto_familiar`, `urgencia`).
- **Rule pack v1**: R-24A (protetiva vigente + descumprimento → art. 24-A),
  R-22 (violência + contexto familiar + urgência → art. 22). Versioned data.
- **Verified chains**: a chain is complete only when the conclusion's
  provision exists verbatim in the corpus; broken chains defer. Answer
  renders conclusion + grounded facts + verbatim premise + disclaimer.
- Live example: *"ele descumpriu a medida protetiva"* → R-24A chain citing
  the current art. 24-A text (incl. Lei 14.994/2024 and Lei 15.383/2026
  amendments).

## Known limitations (honest list)

1. ~~Revoked text not flagged~~ **resolved**: extraction now drops
   `line-through`/`s`/`strike`/`del` content; only current text is quoted.
2. Rule coverage is tiny (2 rules); growth is human-reviewed curation.
3. Citation graph / temporal reasoning / analogy retrieval: not yet built.

## Three levels (build in this order)

| Level | What | Cost | Status |
| ----- | -------------------------------------------- | ------------ | ------------- |
| 1 | Symbolic: syllogism, checklists, citation graph, temporal logic | CPU-free, deterministic | design now |
| 2 | Orchestration: fixed multi-step pipelines of deterministic operations | CPU-free | design now |
| 3 | Tiny CPU LLM proposes, symbolic layer verifies | CPU inference, cached | later, gated |

## Level 1 — the legal syllogism as an engine

```
major premise:  statute text (retrieved, cited)     → art. 22, Lei 11.340/2006
minor premise:  facts extracted from query/document → medida protetiva vigente + descumprimento
conclusion:     rule fires → applicable provision   → art. 24-A (crime de descumprimento)
```

Typed contract (draft, future `reason/` module):

```
Premise:  { ref: "Lei 11.340/2006, art. 22", text: verbatim, origin_ref }
Fact:     { span: "…query text…", field: "medida_vigente", value: true }
Rule:     { conditions: [...], conclusion: "...", refs: [...], version }
Chain:    [Premise | Fact | Rule-application] → conclusion + citations
```

- **Requirements checklists**: e.g. protective measures need urgency + risk
  + domestic context (art. 22). Facts extracted by patterns/lexicon;
  forward-chaining rules fire only when all conditions have cited support.
- **Citation graph**: statute → amendment → súmula → leading case. Graph
  traversal answers "what governs this now, and how is it interpreted?"
- **Temporal reasoning**: law at fact-date vs current law, over versioned
  statute texts. Never silently "update" old excerpts.
- **Analogy retrieval**: similar fact patterns → "analogous precedents",
  presented as citations, never as generated argument.

Rules are **versioned data** (like area packs), human-reviewed, each citing
its source. This is a classical expert system: genuine multi-step reasoning,
fully auditable.

## Level 2 — reasoning as orchestration

A fixed DAG of deterministic steps is itself a reasoning chain:

```
decompose question → identify legal issue → retrieve statute →
retrieve interpreting precedent → check requirements → compose cited chain
```

Multi-hop answers ("art. 22 defines…; Súmula X interprets…; therefore…")
where every hop is a verifiable citation. **Any failed hop → defer.**
Incomplete reasoning never ships.

## Level 3 — tiny LLM as bounded proposer

See [tiny-llm.md](tiny-llm.md). The only safe pattern: LLM *proposes*
(facts, decomposition, draft syllogism), symbolic layer *verifies* against
sources and rules, **verified or discarded**. The three-way gate remains
the final arbiter; the M safety ceiling is untouched.

## Worked example (Maria da Penha)

Query: *"ele descumpriu a medida protetiva, o que acontece?"*

1. Facts extracted: `medida_vigente=true`, `descumprimento=true`
   (spans quoted from the query).
2. Rule R-24A fires: conditions met → conclusion points to art. 24-A,
   Lei 11.340/2006 (pena: reclusão 2–5 anos, redação Lei 14.994/2024).
3. Retrieval supplies the verbatim article text + a published acórdão
   applying it.
4. Answer: conclusion + cited premises + excerpts + disclaimer.
5. `--explain` shows the **whole chain**: facts → rule → premises → scores.

If step 1 cannot ground `medida_vigente` in the query text → chain
incomplete → **defer** with a clarifying question.

## Guarantees

1. Every conclusion carries its full cited chain; no chain, no answer.
2. Rules/facts/premises are typed, versioned, inspectable.
3. The decide gate wraps reasoning: accept/reject/defer applies to chains
   exactly as to classifications.
4. No novel legal argumentation, ever — we connect existing authority,
   we do not invent it.