# Publishing — agavus on Hugging Face

Status: **model repo + Space LIVE**
- Model: https://huggingface.co/Falconiere/agavus-classifier
  (first release 2026-11-24, eval-gated: precision 1.00, M-recall 1.00)
- Demo: https://huggingface.co/spaces/Falconiere/agavus
  (Gradio, cpu-basic; offline bootstrap from bundled public-domain statutes
  in `space/assets/` — UTF-8 transcriptions, LGPD-safe, no network at startup)
Pipeline: `agavus publish` (dry run) / `agavus publish --repo OWNER/NAME`.
HF bucket `Falconiere/agavus` available for private artifacts (corpus
snapshots, future datasets). The **privacy policy governs everything**
([corpus/privacy.md](corpus/privacy.md)). Publishing is **eval-gated**: no
artifact ships without a green eval report (`publish/export.gate`).

## What gets published (three repo types)

| HF repo | Content | When |
| ------------------------------- | ---------------------------------------------- | ------------------------ |
| Model: `Falconiere/agavus-classifier` | vectorizer + logreg + calibrator + label mapping + eval report + model card | after `classify` + evals |
| Space: `Falconiere/agavus` | Gradio demo wrapping `agavus -p`, **CPU-basic tier** (free, 2 vCPU/16GB) | after `answer` works |
| Dataset | ⚠️ mostly **never public** — see below | private only, if ever |

A HF Collection links model ↔ Space ↔ (public) gold sets.

## Privacy rules (non-negotiable)

1. **TJDFT-derived data never goes public** — corpus, holdouts, processed
   records: private HF dataset repos at most, or stay local. Public
   publication would violate the corpus privacy policy (re-identification
   risk survives anonymization).
2. **Safe to publish**: authored gold sets (synthetic, no personal data),
   TPU mapping table, lexicons, code, statute texts (public domain in BR).
3. Every upload passes the same privacy scan as ingestion (zero tolerance).
4. Model cards state the data provenance and the privacy policy explicitly.

## Mechanics

```sh
uv add huggingface_hub          # dep added only when implemented (deptry gate)
uv run huggingface-cli login    # write-scope token

# model repo + artifacts
uv run python -c "from huggingface_hub import create_repo; \
  create_repo('Falconiere/agavus-classifier', repo_type='model')"
uv run huggingface-cli upload Falconiere/agavus-classifier ./artifacts .

# Space (Gradio = pure Python, no Node; auto-builds from requirements.txt)
uv run python -c "from huggingface_hub import create_repo; \
  create_repo('Falconiere/agavus', repo_type='space', space_sdk='gradio')"
```

- HF repos are git repos (LFS): `upload_folder` or git push both work.
- Space README YAML: `sdk: gradio`, `hardware: cpu-basic` — CPU-only,
  consistent with the engine. Tiny-LLM demo (later) needs a paid tier or
  stays local; the classical demo is free-tier friendly.
- Gradio UI mirrors the CLI contract ([cli.md](cli.md)): answer / clarify /
  refuse, citations, disclaimer, `--explain` panel.

## Model card discipline (our design already produces it)

| Card section | Source in this repo |
| ------------------ | ------------------------------------------------ |
| Metrics | eval reports: F1, M recall, ECE, precision/coverage ([evals.md](evals.md)) |
| Provenance | corpus window 2020→date, TJDFT scope, manifest version pinned to artifact |
| Intended use | routing/retrieval aid for F/M; **not legal advice** |
| Limitations | TJDFT-only, lexical core, defer behavior, M↔F overlap |
| Safety | M ceiling, privacy policy, hallucination guards |
| License | code MIT (LICENSE in repo); tiny-LLM artifacts inherit base license (Qwen: Apache-2.0) |

Artifacts are versioned triples: **corpus version + model version + eval
report**. A card without its eval report is incomplete and must not ship.

## Release sequence

1. **Now**: nothing publishable (scaffold). Optionally reserve names as
   empty private repos.
2. **After classify + evals**: model repo (artifacts + report + card).
3. **After answer/CLI**: public Space demo + Collection.
4. **Datasets**: private only; public = synthetic gold sets at most.
5. **Later**: GitHub Actions release workflow — HF token as secret,
   publishes only on tagged releases with green evals (regression policy
   from [evals.md](evals.md) applies to publishing too).

## Rules

1. No publish without a green, versioned eval report.
2. No real case data in any public repo — the git guard principle extends
   to HF uploads (scan before upload).
3. Disclaimer + intended-use + limitations on every public artifact.
4. Reproducibility: published artifact ↔ corpus manifest ↔ eval report
   versions must match exactly.