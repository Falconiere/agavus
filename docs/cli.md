# Agavus CLI

Invoke commands from the repository root with `uv run agavus`. Running without arguments
prints a greeting. The [quickstart](../README.md#run-locally) builds a public demo corpus.

| Command or option | Behavior |
| --- | --- |
| `-p "pergunta"` | Routes a question by meaning. Accepted answers include cited sources. Requires the encoder installed by `embed`. |
| `classify "texto"` | Runs the stored TF-IDF document classifier on one positional text argument. |
| `search "consulta" --limit 5` | Returns lexical hits with a score, reference and snippet. |
| `ingest --source statutes` | Fetches the configured Planalto statutes. For a local copy, add `--raw-file PATH --statute-name NAME`. |
| `ingest --source datajud --limit 500 --pages 1` | Reads TJDFT case metadata. Limit sets the requested page size and pages bounds work in this invocation. |
| `ingest --source djen --pages 1` | Reads published decisions by subject, resuming from the saved position. |
| `label` | Applies the area labeling rules and reports the number needing review. |
| `train` | Fits the document classifier, calibrates it and saves its holdout report. |
| `embed` | Downloads the encoder and indexes the statutes in the selected data directory. Rerun after changing them. |
| `eval` | Evaluates the document classifier on the authored routing gold set. |
| `eval --retrieval` | Evaluates statute ranking on development questions and prints first-hit, top-k and reciprocal-rank metrics. |
| `publish` | Evaluates and stages release files locally. See [publishing](publishing.md) for bootstrap and upload options. |

## Reading the output

`classify --json` returns `probabilities` and `decision`. With `--explain` it adds a
`trace` with probabilities, expected losses, routing mode and the highest-contributing
terms. A successful classification exits with 0 for every decision.

Pass the question immediately after `-p` when requesting JSON output.

```sh
uv run agavus -p "O que diz o art. 22 da Lei 11.340/2006?" --json --explain
```

Question JSON contains `decision`, `area`, `confidence`, `text` and `sources`.
The area is null for defer or reject. Each accepted source has a `ref`.
Confidence is the largest routing probability, or zero if routing was unavailable.
With routing available, `--explain` includes the trace with routing mode `meaning`
and an empty list of lexical contributions. A question exits with 0 for an accepted
answer, 10 for deferral and 20 for rejection.

`search --json` returns a `hits` array. `eval --json` returns document evaluation
metrics. Retrieval evaluation uses text output. See the [answer contract](response-quality.md)
for the evidence required on acceptance.
