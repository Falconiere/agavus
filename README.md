# Agavus

Agavus classifies Brazilian legal documents and answers questions in Portuguese using cited
excerpts. Its production scope covers family law and domestic violence under the Lei Maria
da Penha in the TJDFT. General criminal matters fall outside these two areas.
[Public demo](https://huggingface.co/spaces/Falconiere/agavus) ·
[Document classifier](https://huggingface.co/Falconiere/agavus-classifier)

| Entry point | What it does |
| --- | --- |
| `agavus classify TEXT` | Classifies a document by area with TF-IDF and calibrated logistic regression. Returns area probabilities and an accept, defer or reject decision. |
| `agavus -p QUESTION` and the Space | Route a question with a multilingual E5 encoder and a calibrated linear classifier. An accepted question can return a rule with its supporting statute or a cited excerpt from the corpus. |
| `agavus search QUERY` | Searches indexed texts directly using SQLite FTS5, BM25 and citation matches. |

The decision rule weighs the cost of each error. Misrouting a domestic-violence matter to
family law costs 10 and rejecting it costs 8. A question can still be deferred when the
router accepts its area but the engine cannot support a cited answer.

## Run locally

Install Python 3.12 or later and uv. From the repository root, use a separate directory
for the public demo corpus so that the bootstrap uses only the bundled statutes and
authored examples.

```sh
make install
export AGAVUS_DATA_DIR=data/public-demo
uv run agavus publish --bootstrap space/assets
uv run agavus classify "ele descumpriu a medida protetiva" --explain
uv run agavus search "medida protetiva art. 22"
uv run agavus embed
uv run agavus -p "Qual é o artigo da lei de alienação parental?"
```

The bootstrap builds and evaluates the document classifier locally. The first `embed`
downloads the pinned encoder and indexes the statute articles on CPU. Without the encoder,
`-p` defers. The lexical search and document classifier run independently.

## Data and research

Real case material belongs in the private data directory. The public Space uses bundled
statutes and authored examples. Private decision-reading studies may compare TJDFT, TJSP
and TJRJ with provenance for each court. Classification probabilities describe the routing
area. Estimating the chance that a protective order will be revoked requires a separate
dataset of requests and linked outcomes. The engine has no validated predictor for that task.

## Documentation

- [CLI and exit codes](docs/cli.md) · [Configuration defaults](.env.example)
- [Design and safety requirements](docs/DESIGN.md) · [Extending the area registry](docs/extensibility.md)
- [Corpus sources, privacy and storage](docs/corpus/README.md)
- [Classification and retrieval math](docs/math/README.md) · [Evaluation results and release gates](docs/evals.md)
- [Supported reasoning rules](docs/reasoning.md) · [Answer checks](docs/response-quality.md)
- [Private classifier experiment](docs/math/learned-classifier.md) · [Isolated language-model lab](docs/llm-lab.md)
- [Publishing](docs/publishing.md) · [Contributor instructions](AGENTS.md) · [Commit conventions](docs/commits.md)

Run `make check` for lint, formatting, strict types, duplication, dependencies, size limits
and tests. The implementation and its domain tests live under `src/agavus/`.
