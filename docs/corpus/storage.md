# Storage — where the corpus lives

## Layout

```
data/                      # gitignored, path via AGAVUS_DATA_DIR (default: data/)
  raw/                     # exactly as fetched, never mutated (bytes only)
    datajud/2026-11-24/…   # payload files; metadata lives in corpus.db
    statutes/…
  processed/
    corpus.db              # SQLite: records, labels, runs, watermarks,
                           # raw audit, FTS5 index, embeddings
  models/
    models.db              # SQLite: trained artifacts (vocab, weights BLOBs,
                           # temperature, training reports)
```

**JSON is not used as a database anywhere.** Remaining files: raw payload
bytes (verbatim audit), `.env` config. All structured state — records,
labels, manifests, raw metadata, model artifacts, indexes — is SQLite.

## Retrieval-ready indexes (inside corpus.db)

- **`texts_fts`** — FTS5 virtual table with native `bm25()`: lexical
  retrieval, zero deps. Kept in sync on insert; `rebuild_index()` backfills.
- **`embeddings(doc_id, model_version, dim, vector BLOB)`** — float32
  vectors keyed by doc + model version (upgrade = re-embed, never mix).
  Brute-force cosine KNN (`search_similar`) — fine at corpus scale;
  `sqlite-vec` extension later only if quantization/scale demands it.
- Hybrid retrieval (BM25 + cosine + RRF) will live entirely in this file.

## Why SQLite for processed data (not JSON files)

Access patterns decide the store: ingestion dedup (point lookups by id),
training (bulk scans), labeling/evals/review (queries by area, organ, date,
review flag). File-per-record JSON degrades at the 2020→today window scale
(hundreds of thousands of records): O(n) file opens, no queries, check-then-
write dedup. SQLite (stdlib — **zero new dependencies**) gives:

- atomic dedup: `INSERT OR IGNORE` on the id primary key
- indexed queries: review queue = `SELECT … WHERE needs_review = 1`
- cursor scans for training; single-file backup; WAL; ACID
- schema versioning via `PRAGMA user_version`

Trade-off accepted: not human-diffable — irrelevant because data never
enters git; inspect via `sqlite3` CLI or a future `agavus export`.

## Schema (v1)

```sql
cases(id PK, tribunal, orgao_julgador, classe_tpu, assuntos_tpu JSON,
      data_ajuizamento, movimentos JSON, domain_label, label_source, raw_ref)
texts(id PK, source, origin_ref, published_at, text, citations JSON,
      domain_label, label_source, raw_ref)
labels(record_id PK, area, source, needs_review, reason)   -- derived, re-runnable
meta_runs(run_at, source, fetched, new, window_start, window_end)
meta_watermarks(source PK, value)
raw_payloads(ref PK, source, run, url, method, payload_json, fetched_at)
texts_fts(doc_id UNINDEXED, text)                          -- FTS5 + bm25()
embeddings(doc_id, model_version, dim, vector BLOB)
indexes: cases(orgao), cases(data_ajuizamento), labels(needs_review, area)
```

Model artifacts (`models.db`): `models`, `model_classes`, `model_vocab`,
`model_weights` (BLOBs), `train_reports` — the quality memory of every run.

Records are **immutable once ingested**; the labeling pass writes the
derived `labels` table (upsert), never edits records — provenance preserved.

## Principles

1. **Never in git.** Public repo + LGPD. `.gitignore` covers `data/`, and a
   pre-commit hook fails if anything under `data/` is ever staged — a hard
   guard, not a convention.
2. **Rebuildable cache, not an asset.** Everything in `data/` can be
   regenerated from public sources by re-running ingestion. Losing it costs
   a re-fetch, nothing more.
3. **Raw is sacred.** Payloads stay verbatim files (audit trail); the db is
   derived. Any processed state is explainable via runs + `raw_ref` pointers.
4. **Configurable location.** `AGAVUS_DATA_DIR` (isolated settings module)
   lets the corpus live on a big disk or `~/.cache/agavus` without code
   changes.

## What IS in git

- Ingestion/processing code and the typed schema.
- TPU mapping table and labeling rules (small, public taxonomy).
- Synonym/abbreviation lexicon.
- **Synthetic** test fixtures only.

## Later (only if needed)

- `agavus export --format jsonl|parquet` for training snapshots and private
  HF datasets (columnar export, not the working store).
- DuckDB for heavy eval analytics if SQLite queries stop sufficing.
- Sharing/versioning: object storage (S3/R2) or a private dataset host,
  referenced by config — never git.