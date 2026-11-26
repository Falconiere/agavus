# Publishing Agavus

The release workflow produces a Python package, a Hugging Face model repository with
the document classifier, and a Gradio Space that routes questions by meaning. The Space
bootstraps its own models and corpus from bundled public statutes and authored examples.
[Model repository](https://huggingface.co/Falconiere/agavus-classifier) ·
[Space](https://huggingface.co/spaces/Falconiere/agavus)

## Local staging

Use a fresh data directory. The bootstrap refuses an existing corpus containing case
records or non-statute texts. It imports the bundled articles, labels them, trains the
document classifier and runs the release evaluation.

```sh
AGAVUS_DATA_DIR=data/public-demo uv run agavus publish --bootstrap space/assets
```

Without upload options, the command writes `models.db`, `README.md` (the model card)
and `eval_report.json` to `hf-staging/` under the selected data directory. The model
gate requires M recall of at least 0.98, perfect precision among accepted gold items
and at least one acceptance. Publishing the Space adds a check of its question router
on gold and development queries. [Gate details](evals.md)

## CI/CD

Pull requests, pushes to main and manual CI runs check the tracked-data guard, commit
conventions, `make check` and the offline public bootstrap. The release workflow builds
packages, stages the public artifacts, and uploads according to its trigger and options.
Version tags must match the package version. Manual dispatch with `dry_run=true` runs
staging without publishing to Hugging Face.

For an authorized release, follow the repository's [release procedure](../.toolu/skills/release-agavus/SKILL.md).
The workflow passes `--repo`, `--space` and `--revision` to the CLI and uses the
`HF_TOKEN` secret. Space staging pins the installed Agavus revision. A tagged release
also attaches packages, model database, card and report to the GitHub Release.
[CI workflow](../.github/workflows/ci.yml) · [Release workflow](../.github/workflows/release.yml)

Keep case material, holdouts and private learned weights out of public repositories
and staging directories. The release scanner checks for recognized personal-data patterns.
Use only approved public inputs even when the scan passes. [Corpus privacy](corpus/privacy.md)
