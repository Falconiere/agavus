# Corpus ingestion

`agavus ingest` accepts three sources. DataJud supplies TJDFT metadata, DJEN supplies
published decision texts and Planalto supplies statutes. Local statute payloads can also
be imported. Source selection, limits and import flags are documented in the [CLI](../cli.md).

1. The fetcher stores received payloads under `raw/` and records request metadata in
   SQLite. Processed records carry a `raw_ref`. Source parsers select fields, extract
   citations and apply the relevant text redactions. Raw payloads remain private.
2. The store inserts cases by case identifier and texts by a hash of source and public
   reference. Replaying the same key does not create a second row. Different references
   to the same ruling can still produce different records.
3. DataJud and DJEN save progress as they traverse pages. DataJud walks months and uses
   a search cursor inside each month. DJEN tracks pages separately for each subject term.
   A failed request can leave earlier successful pages stored. [Date range and stopping rules](data-range.md)
4. Run `agavus label` to write derived area labels and review flags. Run `agavus embed`
   after changing statutes to refresh the semantic index. Lexical indexing is maintained
   on text insertion. [Labeling](labeling.md) · [Storage](storage.md)
