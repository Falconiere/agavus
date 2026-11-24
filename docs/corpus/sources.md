# Sources — where the data comes from

All case sources are filtered to **TJDFT** and to the F/M subject, with
no start date ([data-range.md](data-range.md)): today backward until empty.

| Source | Content | Role | Access |
| -------------------------- | -------------------------------------- | ---------------------- | ---------------------------- |
| DataJud API (CNJ) | case metadata + TPU classe/assuntos | labels, features | public API, filter `tribunal = TJDFT` |
| DJEN | full-text decisions, national diary | training text, retrieval | filter by tribunal + date |
| TJDFT jurisprudence portal | acórdãos/decisões search | retrieval supplement | public web search |
| Planalto / LexML | statutes: Lei 11.340/2006, 12.318/2010, 13.058/2014, 14.994/2024, CC arts. 1.511-1.783 | retrieval backbone | public, **not windowed** |

## What each contributes

- **DataJud**: the label backbone. TPU codes + órgão julgador give F/M
  labels without reading a single full text. Also dates and movements for
  features and time splits. No full text — and that is fine.
- **DJEN**: the language backbone. Published decisions are the text the
  classifier learns from and the excerpts answers quote. Every item carries
  a canonical reference (órgão, relator, publication date) → `origin_ref`.
- **Portal**: fills gaps where DJEN coverage is thin; same schema, same
  provenance rules.
- **Statutes**: current consolidated texts; articles are the retrieval units
  (an answer cites "art. 22, III, Lei 11.340/2006", not a whole law).

## Fetching rules

1. **Politeness**: rate-limit requests, identify the client, respect
   robots/terms; back off on errors. Public APIs are a shared resource.
2. **Raw preservation**: store every response as fetched under `data/raw/`
   (with fetch timestamp + query parameters). Processing never mutates raw.
3. **Resume, don't restart**: watermarks per source make runs incremental
   and crash-safe (see [ingestion.md](ingestion.md)).
4. **Provenance**: every processed record points back to its raw payload and
   its public reference (`origin_ref`).

## Access spike findings (2026-11-24)

- **DataJud**: public APIKey auth works; the cluster is frequently
  overloaded — HTTP 429 (`es_rejected_execution_exception`), queries up to
  ~60s, and 200 responses with **empty hits**. An ISO range on
  `dataAjuizamento` returned `hits.total = 0` (shards skipped). A compact
  `YYYYMMDDHHmmss` bound reaches the index. Sorting on `numeroProcesso`
  is HTTP 400 (text field, fielddata disabled); the tie-break is `_id`.
  A `match_phrase` subject query was accepted far enough to hit the search
  pool (429, not 400) on 2026-11-24; the cluster was too busy to return
  hits. That 429 is the search thread pool rejecting work
  (`es_rejected_execution_exception`), not a quota header to route around.
  The [terms of use](https://datajud-wiki.cnj.jus.br/api-publica/termo-uso)
  forbid abusive use. Handling in `corpus/fetch.py` and `corpus/datajud.py`:
  one request at a time, 2s minimum interval, Retry-After honored, equal-jitter
  backoff capped at 120s, 8 retries, 90s timeout. Each search is one calendar
  month (`gte`/`lte` compact) plus the subject `match_phrase` filter, with
  `search_after` inside the month. A genuine empty month moves backward.
  Thirty-six empty months after a stored hit end the walk. A body without
  `total = 0` does not. The cursor stays put when retries run out, so the
  next invocation resumes.
- **Planalto**: reachable via normal networks; direct egress from the dev
  sandbox timed out (parser validated against real pages fetched via
  firecrawl). Statute ingestion is implemented and fixture-tested; live runs
  depend on network egress.
- **DJEN**: `GET https://comunicaapi.pje.jus.br/api/v1/comunicacao` returns
  HTTP 403 from this host (CloudFront country block). The client is
  implemented and fixture-tested. Page 1 is treated as the newest page.
  `itensPorPagina` is capped at 50. Intimações, citações, editais, and
  pautas are not stored. A live walk needs a Brazilian IP.

## Known coverage asymmetry (segredo de justiça)

Family (F) and Maria da Penha (M) full texts are mostly sealed:

| Domain | Metadata (DataJud) | Full text (DJEN/portal) |
| ------ | ------------------ | ------------------------------------------ |
| F | available | scarce |
| M | available | scarce (published acórdãos exist) |

General criminal full texts are abundant but **out of scope**. M's criminal
facet (lesão corporal, ameaça, feminicídio, art. 24-A) is covered by published
acórdãos from criminal turmas and the specialized varas.

Consequence: F/M classification leans on metadata + citation patterns;
F/M retrieval leans on statutes + published acórdãos. This is a design
fact, not a bug to fix. See [privacy.md](privacy.md).