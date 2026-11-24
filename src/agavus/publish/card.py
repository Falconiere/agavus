"""Model card generation from real metrics and provenance (docs/publishing.md).

Every card section is fed by measured data — no hand-waved numbers.
"""

from agavus.classify.pipeline import TrainReport
from agavus.evals.run import EvalReport

DISCLAIMER = "This is not legal advice. Always consult a qualified lawyer."


def build_model_card(
  *,
  train_report: TrainReport,
  eval_report: EvalReport,
  corpus_stats: dict[str, int],
  gold_version: str,
) -> str:
  """Render the HF model card (README.md) for the classifier artifact."""
  stats_rows = "\n".join(f"| {name} | {count} |" for name, count in sorted(corpus_stats.items()))
  return f"""---
license: mit
language:
- pt
tags:
- legal
- text-classification
- brazilian-law
- portuguese
pipeline_tag: text-classification
---

# agavus-classifier

CPU-only legal text classifier for Brazilian law, scoped to **TJDFT** and two
routing domains: **F** (direito de família) and **M** (Maria da Penha /
violência doméstica). Everything else is **O** (out of scope).

Part of the [agavus project](https://github.com/Falconiere/agavus): a
calibrated classifier + three-way decision engine (accept / reject / defer)
with cited, extractive answers. **No GPU, no LLM in the core.**

## Intended use

Route legal queries/documents to the correct domain with honest confidence;
defer instead of guessing. Answers in the full engine are verbatim cited
excerpts, never generated opinions. {DISCLAIMER}

## Architecture

TF-IDF (word 1-2 grams, char 3-grams, negation-scoped features, decision
verbs, citation flags) → multinomial logistic regression (deterministic
full-batch GD, class-balanced) → temperature scaling → cost-sensitive
three-way decision (loss matrix with a safety ceiling: misrouting
Maria da Penha as family law costs 10; rejecting it costs 8).

## Metrics (measured, not claimed)

Gold set `{gold_version}` (10 authored synthetic queries, firewalled from
training):

| Metric | Value |
| ---------------------------- | ----- |
| Accuracy | {eval_report.accuracy:.2f} |
| Macro-F1 | {eval_report.macro_f1:.2f} |
| M-recall (safety ≥ 0.98) | {eval_report.m_recall:.2f} |
| Precision on accepted | {eval_report.precision_on_accepted:.2f} |
| Coverage | {eval_report.coverage:.2f} |
| Accept / defer / reject | {eval_report.accepted}/{eval_report.deferred}/{eval_report.rejected} |

Holdout at train time: {train_report.n_train} train / {train_report.n_holdout}
holdout · acc {train_report.holdout_accuracy:.2f} · ECE {train_report.holdout_ece:.3f}
· temperature {train_report.temperature:.2f}.

## Training data provenance

| Source | Documents |
| ------ | --------- |
{stats_rows}

Public-domain Brazilian statutes (Planalto consolidated texts): Lei
11.340/2006, Lei 12.318/2010, Lei 13.058/2014, Lei 14.994/2024, Civil Code
arts. 1.511-1.783 (família book), plus authored synthetic seed queries.
**No real case data, no personal data** (LGPD). TJDFT case-metadata ingestion
(DataJud) is implemented but not included in this artifact.

## Limitations

- Statute-language domain: colloquial query coverage is partial; the decision
  gate defers low-confidence inputs instead of guessing.
- Two routing areas only; other legal areas are out of scope by design.
- Lexical core: paraphrases outside the vocabulary may be missed.
- Not a substitute for professional legal judgment.

## Safety & privacy

- Safety ceiling enforced in code and tests; no threshold weakening.
- Eval-gated releases: this card ships only with a passing gold evaluation.
- LGPD: no personal data in training, artifacts, or outputs.

## How to use

```sh
git clone https://github.com/Falconiere/agavus && cd agavus
make install
uv run agavus train
uv run agavus -p "ele descumpriu a medida protetiva, o que acontece?"
```

The artifact `models.db` (SQLite) holds vocabulary, weights, temperature,
and training reports; load it with `agavus.classify.store.load_model`.

## License

MIT. {DISCLAIMER}
"""
