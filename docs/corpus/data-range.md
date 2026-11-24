# Data range — current date back to the oldest available row

## The window

There is no start date. Ingestion starts at the current date and walks
backward, newest first, until the source returns nothing more for the
subject (família and Maria da Penha at the TJDFT).

`AGAVUS_DATA_START` remains in settings so older environment files still
load. Fetches do not read it.

## Why there is no floor

The subject is the area, not a reform year. Older F/M metadata and published
decisions stay in scope. Lei 14.550/2023 and Lei 14.994/2024 still matter;
they are not a reason to drop earlier cases. Volume is bounded by the
subject filter and by the 2s request throttle, not by a calendar cutoff.

## Walk

- Sort newest → oldest. A short page, or an empty continuation after a
  saved cursor, finishes that source.
- The cursor is committed after every saved page. A 429, a 504, or a
  timed-out empty body does not move it.
- A finished walk only asks for rows newer than the newest hit already
  stored, still bounded by the end of today.
- Re-running a finished day adds only that new head. Ids are deterministic.

## Exception: statutes are NOT windowed

Statutes enter as the **current consolidated text** (texto compilado,
Planalto/LexML), regardless of the case window.

Temporal caveat: a 2021 decision cites the law *as it was in 2021*. Retrieval
must prefer current text for answers, and cited excerpts from older decisions
keep their original wording with their date visible — never silently
"updated". Amendment awareness is a later refinement.

## Splits and snapshots

- **Train/held-out split by case, preferably by time** (latest year held
  out) so evaluation mimics "future cases" and prevents leakage. The split
  is not a fetch cutoff.
- Each ingestion run records a **manifest** (run date, window, counts per
  source, watermark) so any corpus state is explainable and reproducible.