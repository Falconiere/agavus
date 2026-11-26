# Corpus sources

| CLI source | What enters the store | Use |
| --- | --- | --- |
| `datajud` | TJDFT process metadata, including subject codes, issuing court and available dates | Case labeling and provenance |
| `djen` | Published TJDFT decision texts found through the configured family-law and domestic-violence terms | Lexical retrieval and eligible labeled document training |
| `statutes` | Articles parsed from configured Planalto texts or supplied local statute files | Cited answers, rule premises, retrieval and document training |

The production CLI implements these three adapters. The record schema also accepts a
`portal` source for provenance. The private reading studies use court portals and retain
their own references. Their files do not automatically enter the production index.

DataJud queries use month ranges, compact filing-date bounds and a `search_after`
cursor. The client limits request frequency and retries transient errors. DJEN keeps
each term's position separately. Network errors and inaccessible material leave gaps
in coverage. Published decisions cannot establish how many requests a court received
or how many are still pending. [Range](data-range.md) · [Ingestion](ingestion.md) · [Privacy](privacy.md)
