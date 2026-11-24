# Privacy — LGPD, anonymization, segredo de justiça

The repo is **public**. Real case data must never be committed, even
anonymized (re-identification risk). Privacy is enforced at ingestion,
before any processing or storage of processed records.

## What is stripped (at ingestion, before anything else)

| Category | Examples | Action |
| ------------------------ | ------------------------------------ | ------------------------- |
| Party names | authors, réus, witnesses, minors | remove / replace with role tokens |
| Personal documents | CPF, RG, CNH numbers | remove |
| Contact / location | addresses, phones, emails | remove |
| Minor identifiers | anything identifying children | remove (heightened protection) |

Role tokens keep text usable: "a autora", "o réu" survive; names do not.
Automated scans verify no leakage (design doc §8: zero tolerance).

## What stays (public by law)

- Process numbers, classes, TPU codes, órgãos, dates, movements — public
  metadata via DataJud.
- Published decision text (acórdãos/decisões in DJEN) — public by
  publication, after anonymization of any residual personal data.
- Statutes — fully public.

## Segredo de justiça (sealed cases)

Family (F) and Maria da Penha (M) cases are frequently sealed:

- **Never attempt to access sealed full texts.** No scraping around
  access controls, no exceptions.
- Metadata from DataJud is public and sufficient for labeling/features.
- Retrieval for F/M leans on statutes + *published* acórdãos.
- Coverage asymmetry (F/M metadata-heavy, P text-heavy) is a design fact —
  see [sources.md](sources.md).

## Storage consequences

- Real data only in gitignored `data/` ([storage.md](storage.md)), with a
  pre-commit guard against staged `data/` files.
- Test fixtures in git are **synthetic** — invented records, never edited
  real ones.
- If sharing is ever needed: private object storage / private dataset host,
  referenced by config — never git, never public.