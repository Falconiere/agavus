# Storage — where the corpus lives

## Layout

```
data/                      # gitignored, path via AGAVUS_DATA_DIR (default: data/)
  raw/                     # exactly as fetched, never mutated
    datajud/2026-11-24/…   # response payloads + query params + fetch time
    djen/2026-11-24/…
    statutes/…
  processed/               # cleaned, anonymized, labeled, typed records
    cases/…
    texts/…
    manifest.json          # run date, window, counts per source, watermark
```

## Principles

1. **Never in git.** Public repo + LGPD. `.gitignore` covers `data/`, and a
   pre-commit hook fails if anything under `data/` is ever staged — a hard
   guard, not a convention.
2. **Rebuildable cache, not an asset.** Everything in `data/` can be
   regenerated from public sources by re-running ingestion. Losing it costs
   a re-fetch, nothing more. This is why no backup machinery is needed for
   the MVP.
3. **Raw is sacred.** Processed records are versions derived from raw;
   reprocessing never edits raw. Any processed state is explainable via its
   manifest + `raw_ref` pointers.
4. **Configurable location.** `AGAVUS_DATA_DIR` (isolated settings module)
   lets the corpus live on a big disk or `~/.cache/agavus` without code
   changes.

## What IS in git

- Ingestion/processing code and the typed schema.
- TPU mapping table and labeling rules (small, public taxonomy).
- Synonym/abbreviation lexicon.
- **Synthetic** test fixtures only.

## Later (only if needed)

- Sharing/versioning: object storage (S3/R2) or a private dataset host,
  referenced by config — never git.
- DVC-style snapshot versioning only if reproducibility demands it; skip
  for MVP (manifests already explain every corpus state).