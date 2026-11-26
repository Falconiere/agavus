# Private corpus and model storage

The configured `AGAVUS_DATA_DIR` defaults to `data/` and contains the following stores.

| Path relative to the data directory | Contents |
| --- | --- |
| `raw/<source>/<run>/` | Source payloads. The corresponding request metadata is recorded in the corpus database. |
| `processed/corpus.db` | Cases, texts, derived labels and review flags, raw references, ingestion runs and watermarks. |
| `models/models.db` | Document classifier vocabulary, weights, classes, temperatures and training reports. The question router caches its fitted parameters by fingerprint. |
| `models/encoder/` | Pinned ONNX encoder and tokenizer for CPU question routing and statute retrieval. |
| `experiments/` | Isolated private research evidence, snapshots and reports. |

Text insertion updates the FTS5 index `texts_fts`. BM25 weights the public reference
and popular statute names alongside the text. The `embeddings` table stores float32
vectors keyed by chunk identifier and encoder version. `agavus embed` indexes articles
in chunks of up to 120 words. [Corpus schema and indexes](../../src/agavus/corpus/db.py)

Input records use deterministic identifiers. Re-running labeling updates the derived
labels separately. A source can change or cease to be publicly available, so retain
the original private evidence and its provenance when reproducibility matters.
[Ingestion](ingestion.md) · [Privacy](privacy.md)
