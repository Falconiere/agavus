# Data range and continuation

TJDFT family-law and domestic-violence material is in scope across all available dates.
`AGAVUS_DATA_START` is loaded for compatibility. Ingestion applies no fixed starting year.

- **DataJud** starts in the current month, reads pages using `search_after`, then
  works backward by calendar month. After a stored hit, 36 consecutive empty months
  end the historical walk. A run without any hits can stop with a saved position for a
  later retry. A completed walk refreshes the head of the series.
- **DJEN** walks pages from newest to oldest for each configured term. A short page
  completes that term. A later run rechecks the first page and may resume the walk
  if a full page contains only new eligible decisions.
- **Statutes** use the article text supplied by Planalto or a local source file.
  The case date range does not filter articles. An answer quotes the locally stored
  version. The engine does not reconstruct the law in force at a historical fact date.

Source access, term filters and stopping rules affect coverage. Training holds related
cases and duplicate text together independently of these fetch boundaries.
[Ingestion](ingestion.md) · [Training splits](../evals.md#splits-and-leakage)
