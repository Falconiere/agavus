# ⚖️ agavus

**CPU-only legal intelligence for Brazilian law** — classification + cited
retrieval, no GPU, no Node. Built on a domain-organized Python scaffold with
strict, machine-enforced quality gates.

![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)
![uv](https://img.shields.io/badge/managed-uv-6f42c1)
![Ruff](https://img.shields.io/badge/lint%20%2B%20format-ruff-d7ff64)
![mypy](https://img.shields.io/badge/types-mypy%20strict-blue)
![pre-commit](https://img.shields.io/badge/hooks-pre--commit-fab040)

> Status: **design complete, implementation not started.**
> Full design: **[docs/DESIGN.md](docs/DESIGN.md)**

## 🎯 Vision

A legal **classifier + search/answer engine** restricted, for now, to **TJDFT**
data and three routing domains:

| Code | Domain | Examples |
| :--: | ------------------ | ---------------------------------------------- |
| **F** | Direito de família | divórcio, guarda, alimentos, união estável |
| **M** | Maria da Penha | Lei 11.340/2006, medidas protetivas |
| **P** | Direito penal | crimes em geral, dosimetria, execução penal |

Answers are **cited excerpts from statutes and decisions — never generated
opinions**. When uncertain, the system *defers*; it does not guess.

## 🏗️ Architecture

```mermaid
flowchart LR
  subgraph sources["🌐 Public sources · TJDFT"]
    DJ["DataJud API<br/>metadata + TPU codes"]
    DJEN["DJEN<br/>full-text decisions"]
    LAW["Planalto / LexML<br/>statutes"]
  end

  subgraph engine["🐍 Engine (in git: code only)"]
    ING["corpus<br/>ingest + anonymize"]
    TAX["taxonomy<br/>TPU map + weak labels"]
    CLF["classify<br/>TF-IDF + calibrated logreg"]
    DEC["decide<br/>three-way gate"]
    RET["retrieve<br/>BM25 + citation features"]
    ANS["answer<br/>extractive + cited"]
  end

  DATA[("🗄️ data/ · gitignored<br/>raw + processed")]

  DJ --> ING
  DJEN --> ING
  LAW --> ING
  ING <--> DATA
  ING --> TAX --> CLF --> DEC
  DEC -->|accept| RET --> ANS
  DEC -->|defer| HUM["clarify / human review"]
  DEC -->|reject| OOS["out of scope"]
```

Key ideas:

- 🧮 **Mathematical core, CPU-trainable**: TF-IDF + calibrated logistic
  regression; BM25 + statute-citation features. Small CPU embeddings and a
  quantized CPU LLM are optional later layers — never the source of truth.
- 🏷️ **Free labels**: CNJ TPU taxonomy + DataJud metadata; TJDFT specialized
  varas as weak labels.
- 🔀 **Routing taxonomy, not legal branches**: Maria da Penha is a hybrid
  statute (penal + civil/família); M↔F overlap is expected and handled by the
  boundary region, with multi-label as a documented evolution path.
- 🔒 **Corpus never in git** (public repo + LGPD): real data lives in a
  gitignored `data/` dir (`AGAVUS_DATA_DIR`); the repo holds only code,
  schema, taxonomy tables, and synthetic fixtures.

## 🔀 Three-way decisions

Cost-sensitive accept / reject / defer (Yao's decision-theoretic rough sets).
Safety-critical errors are priced highest — e.g. misrouting Maria da Penha as
ordinary family law (loss 10) or silently dropping it (loss 8).

```mermaid
flowchart TD
  X["calibrated probabilities<br/>P(F), P(M), P(P), P(O)"] --> L["expected loss per action<br/>Σ λ(action | state) · P(state | x)"]
  L --> A{"minimum-loss<br/>action"}
  A -->|accept| ACC["✅ classify / answer<br/>with citations"]
  A -->|reject| REJ["🚫 out of scope — say so"]
  A -->|defer| DEF["🤔 clarify → re-retrieve<br/>→ human review queue"]
  DEF -.->|new evidence| X
```

| Domain | α (accept ≥) | β (reject ≤) | Behavior |
| -------------- | :----------: | :----------: | ------------------------------------- |
| Família | ~0.86 | ~0.25 | strictest acceptance (M→F catastrophic) |
| Maria da Penha | ~0.80 | **~0.12** | hardest to reject; weak signals defer |
| Penal | ~0.80 | ~0.25 | balanced |

## 🗺️ Roadmap

```mermaid
flowchart LR
  C["1 · corpus"] --> T["2 · taxonomy"] --> K["3 · classify"] --> D["4 · decide"] --> R["5 · retrieve"] --> A["6 · answer"]
  A -.-> NEXT["later: CPU embeddings,<br/>subtopics, other tribunals"]
```

## 🚀 Quickstart

Requirements: Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
make install        # uv sync --extra dev + install pre-commit hooks
cp .env.example .env
make check          # all quality gates + tests
uv run agavus       # -> Hello, world!
AGAVUS_NAME=Brasil uv run agavus   # -> Hello, Brasil!
```

Every commit runs all gates via **pre-commit** (local hooks through `uv`).
Run them manually with `make hooks`.

## 🛠️ Commands

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

## 📁 Structure (by domain)

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
  corpus/                # placeholders — design complete, no implementation yet
  taxonomy/              #   TPU mapping + weak labels
  classify/              #   TF-IDF + calibrated logreg
  decide/                #   three-way decision gates
  retrieve/              #   BM25 + citations
  answer/                #   extractive, cited answers
scripts/
  pyutils.py             # shared tokenization helpers for quality scripts
  check_sizes.py         # size-limit enforcement
  check_duplication.py   # copy-paste detection (jscpd replacement)
  *_test.py              # co-located tests for each script
```

New domains (`corpus`, `taxonomy`, `classify`, `decide`, `retrieve`, `answer`)
follow the same pattern: module + co-located `*_test.py`.

## ⚙️ Configuration

Isolated in `agavus.config.settings`: a frozen, fully typed `Settings`
dataclass loaded from `.env` and environment variables. Copy `.env.example`
to `.env`; never commit `.env`.

| Variable      | Default | Meaning              |
| ------------- | ------- | -------------------- |
| `AGAVUS_NAME` | `world` | Name used in greeting |

Planned (see design doc): `AGAVUS_DATA_DIR` for the gitignored corpus location.

## 📏 Conventions (enforced, not optional)

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

## 🧰 Tooling notes

- Everything runs through `uv`; pre-commit hooks are `repo: local` with
  `language: system`, so no Node and no extra hook environments are needed.
- `uv.lock` is committed for reproducible installs.