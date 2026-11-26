# Corpus documentation

Production ingestion covers TJDFT case sources and the configured Brazilian statutes.
Resolved document labels feed training. Citable references and excerpts feed search
and answers. Private studies across tribunals keep their own provenance and evidence.

| Topic | Where to look |
| --- | --- |
| Available adapters and coverage | [Sources](sources.md) |
| Time window, pagination and stopping rules | [Data range](data-range.md) |
| Fetching, inserting and labeling records | [Ingestion](ingestion.md) |
| Typed records and provenance fields | [Schema](schema.md) |
| Area labels, conflicts and review flags | [Labeling](labeling.md) |
| Handling case material and sealed records | [Privacy](privacy.md) |
| SQLite, indexes and artifacts | [Storage](storage.md) |

Real case data stays outside Git. Training and evaluation keep records from the same
case together. [Production design](../DESIGN.md) · [Training splits](../evals.md#splits-and-leakage)
