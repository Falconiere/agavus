# Data range — 2020-01-01 → current date

## The window

- **Start**: `2020-01-01` (fixed, configurable via `AGAVUS_DATA_START`).
- **End**: the **current date at ingestion time** — rolling, not fixed.
  Example: an ingestion run on 2026-11-24 covers 2020-01-01 → 2026-11-24.
- Applied to case metadata (DataJud) and decisions (DJEN/portal) by their
  distribution/decision dates.

## Why 2020

1. **Legal currency**: the window captures the recent reforms that define
   current practice — Lei 14.550/2023 (autonomous protective measures),
   Lei 14.994/2024 (feminicídio as autonomous crime, art. 24-A crime of
   violating protective measures). Older data would teach outdated procedure.
2. **Digital maturity**: post-2020 records are consistently born-digital
   (PJe/DJEN), with stable metadata and TPU coding.
3. **Volume control**: a single court × ~7 years is CPU-manageable for
   training and indexing; measure actual counts at first ingestion.
4. **Vocabulary stability**: less terminological drift for the classifier.

## Rolling end: incremental ingestion

- Each run fetches only what is **new since the last successful run**
  (watermark stored with the processed data), then re-labels/re-indexes
  affected records. Idempotent: re-running a day changes nothing.
- The corpus is always "2020 → today"; there is no frozen end date.

## Exception: statutes are NOT windowed

Statutes enter as the **current consolidated text** (texto compilado,
Planalto/LexML), regardless of the case window.

Temporal caveat: a 2021 decision cites the law *as it was in 2021*. Retrieval
must prefer current text for answers, and cited excerpts from older decisions
keep their original wording with their date visible — never silently
"updated". Amendment awareness is a later refinement.

## Splits and snapshots

- **Train/held-out split by case, preferably by time** (e.g. train ≤ 2025,
  held-out 2026+) so evaluation mimics "future cases" and prevents leakage.
- Each ingestion run records a **manifest** (run date, window, counts per
  source, watermark) so any corpus state is explainable and reproducible.