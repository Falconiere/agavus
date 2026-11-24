# agavus

Domain-organized Python scaffold (hello world) with strict, enforced quality gates.
Groundwork for a Brazilian legal LLM project. **Pure Python tooling — no Node.**

## Requirements

- Python >= 3.12 and [uv](https://docs.astral.sh/uv/)

## Quickstart

```sh
make install        # uv sync --extra dev + install pre-commit hooks
cp .env.example .env
make check          # all quality gates + tests
uv run agavus       # -> Hello, world!
AGAVUS_NAME=Brasil uv run agavus   # -> Hello, Brasil!
```

Every commit automatically runs all gates via **pre-commit** (local hooks
through `uv`, see `.pre-commit-config.yaml`). Run them manually with
`make hooks`.

## Commands

| Command            | What it does                                            |
| ------------------ | ------------------------------------------------------- |
| `make lint`        | Ruff lint + format check                                |
| `make format`      | Ruff auto-format + lint autofix                         |
| `make typecheck`   | mypy `--strict` (all inputs/outputs typed, no `Any`)    |
| `make dupcheck`    | Token-based clone detector — zero tolerance             |
| `make depcheck`    | deptry — fails on unused/missing dependencies           |
| `make sizes`       | Custom checker: function/file size limits               |
| `make test`        | pytest (tests are co-located with sources)              |
| `make check`       | Everything above, in order                              |
| `make hooks`       | pre-commit against all files                            |

## Structure (by domain)

```
src/agavus/
  __main__.py            # CLI entry point
  config/                # configuration domain (isolated, env-driven)
    settings.py
    settings_test.py     # tests live next to the code they test
  greeting/              # greeting domain (hello world)
    greet.py
    greet_test.py
scripts/
  pyutils.py             # shared tokenization helpers for quality scripts
  check_sizes.py         # size-limit enforcement
  check_duplication.py   # copy-paste detection (jscpd replacement)
  *_test.py              # co-located tests for each script
```

Each domain is a package under `src/agavus/`. New domains (e.g. `corpus`, `training`)
follow the same pattern: module + co-located `*_test.py`.

## Configuration

Configuration is isolated in `agavus.config.settings`: a frozen, fully typed
`Settings` dataclass loaded from `.env` (via python-dotenv) and environment
variables. Copy `.env.example` to `.env`; never commit `.env`.

| Variable      | Default | Meaning              |
| ------------- | ------- | -------------------- |
| `AGAVUS_NAME` | `world` | Name used in greeting |

## Conventions (enforced, not optional)

- **2 spaces** indentation everywhere (`.editorconfig`, Ruff). Makefile recipes
  are the sole exception — Make requires literal tabs.
- **Strict types**: mypy `strict = true`; every function has annotated inputs
  and outputs; no untyped defs, no `Any`.
- **Function <= 80 code lines**, **file <= 500 code lines** (comments and blank
  lines excluded) — enforced by `scripts/check_sizes.py`.
- **No duplication**: `scripts/check_duplication.py` fails on any shared run of
  >= 30 tokens (comments/whitespace ignored), within or across files.
- **No unused dependencies**: deptry checks imports against `pyproject.toml`.
  (knip is JavaScript-only; deptry is its Python equivalent.)
- **Tests co-located**: `foo.py` is tested by `foo_test.py` in the same directory.
- **Domain-first layout**: group by business domain, not by technical layer.
- **All gates run on every commit** via pre-commit local hooks.

## Tooling notes

- Everything runs through `uv`; pre-commit hooks are `repo: local` with
  `language: system`, so no Node and no extra hook environments are needed.
- `uv.lock` is committed for reproducible installs.