# agavus

CPU-only legal intelligence for Brazilian law — starting as a domain-organized
Python scaffold (hello world) with strict, enforced quality gates.
**Pure Python tooling — no Node, no GPU required.**

## Vision

A legal **classifier + search/answer engine** restricted, for now, to **TJDFT**
data and three routing domains:

| Code | Domain |
| ---- | ---------------------------------------------- |
| F | Direito de família (divórcio, guarda, alimentos) |
| M | Maria da Penha (Lei 11.340/2006, medidas protetivas) |
| P | Direito penal / criminal |

Core ideas (full details in **[docs/DESIGN.md](docs/DESIGN.md)**):

- **Mathematical core, CPU-trainable**: TF-IDF + calibrated logistic
  regression for classification; BM25 + statute-citation features for
  retrieval. Answers are *cited excerpts*, never generated opinions.
- **Free labels**: CNJ TPU taxonomy + DataJud metadata; TJDFT specialized
  organs (varas) as weak labels.
- **Three-way decisions** (accept / reject / defer) with a cost-sensitive loss
  matrix — safety-critical errors (e.g. misrouting Maria da Penha as ordinary
  family law) are priced highest; ambiguous cases are deferred, never forced.
- **Routing taxonomy, not legal branches**: Maria da Penha is a hybrid statute
  (penal + civil/família); M↔F overlap is expected and handled by the boundary
  region, with multi-label as a documented evolution path.
- **Corpus never in git** (public repo + LGPD): real data lives in a
  gitignored, configurable `data/` dir (`AGAVUS_DATA_DIR`); the repo holds
  only code, schema, taxonomy tables, and synthetic test fixtures.

Roadmap: `corpus → taxonomy → classify → decide → retrieve → answer`.
Status: **design complete, implementation not started.**

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
docs/
  DESIGN.md              # architecture, data, models, loss matrix, evaluation
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

Each domain is a package under `src/agavus/`. New domains (`corpus`,
`taxonomy`, `classify`, `decide`, `retrieve`, `answer`) follow the same
pattern: module + co-located `*_test.py`.

## Configuration

Configuration is isolated in `agavus.config.settings`: a frozen, fully typed
`Settings` dataclass loaded from `.env` (via python-dotenv) and environment
variables. Copy `.env.example` to `.env`; never commit `.env`.

| Variable      | Default | Meaning              |
| ------------- | ------- | -------------------- |
| `AGAVUS_NAME` | `world` | Name used in greeting |

Planned (see design doc): `AGAVUS_DATA_DIR` for the gitignored corpus location.

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
- **No real case data in git, ever** — synthetic fixtures only; corpus lives
  outside the repo (LGPD).

## Tooling notes

- Everything runs through `uv`; pre-commit hooks are `repo: local` with
  `language: system`, so no Node and no extra hook environments are needed.
- `uv.lock` is committed for reproducible installs.