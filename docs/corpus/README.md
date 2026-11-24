# Corpus documentation

Deep-dive docs for the agavus corpus — the raw material of the whole system.
The corpus is the only part that touches reality: models are interchangeable
math; **clean labels, honest provenance, and verifiable citations are the moat**.

## Two jobs, one collection

| Role | Feeds | Needs |
| ------------------ | ------------------------------ | ---------------------------------------- |
| Training corpus | classifier (TF-IDF + logreg) | texts **with labels** (F/M/P/O) |
| Retrieval corpus | BM25 index → cited answers | statutes + decisions, **searchable**, with source references |

A document can serve both: a labeled acórdão trains the classifier *and* gets
indexed for answers.

## Doc map

| Doc | Question it answers |
| ------------------------------- | ---------------------------------------------- |
| [data-range.md](data-range.md) | Which time window? Why 2020 → current date? |
| [sources.md](sources.md) | Where does data come from, and how do we fetch? |
| [schema.md](schema.md) | What does a record look like, field by field? |
| [ingestion.md](ingestion.md) | How does raw data become processed records? |
| [labeling.md](labeling.md) | Where do F/M/P/O labels come from? |
| [privacy.md](privacy.md) | LGPD, anonymization, segredo de justiça |
| [storage.md](storage.md) | Where does it live, and why never in git? |

## Invariants (never violated)

1. **TJDFT only** for now; other tribunals are a later domain-shift phase.
2. **Window**: 2020-01-01 → current date at ingestion (rolling).
3. Every label carries `label_source`; every text carries `origin_ref`.
4. Party names and personal data stripped at ingestion (LGPD).
5. Real data **never in git**; corpus is a rebuildable cache, not an asset.
6. Same case never appears in both train and held-out splits.

Parent design: [../DESIGN.md](../DESIGN.md) §3–5.