# AGENTS.md

Guide for AI coding agents in this repo. **Phase: MVP implemented and
published** (GitHub + HF model repo + live Gradio Space). Domain packages
under `src/agavus/` are live; `retrieve/reason/answer` work end-to-end.
Implement against the design docs, not from scratch ideas.

## Read before touching

| About to change… | Read first |
| ------------------------------ | -------------------------------------------- |
| anything | `docs/DESIGN.md` (the spec; §6.1 mitigations are **requirements**, not options) |
| corpus / ingestion / data | `docs/corpus/` (range, schema, labeling, privacy, storage) |
| models / features / decisions | `docs/math/` (formulas + worked examples) |
| labels, areas, loss matrix | `docs/extensibility.md` (areas are data packs) |
| GPU rental, machines, disks | `docs/gpu-brasil.md` (read freely; ask before every change) |
| LLM lab | `docs/llm-lab.md` (isolation and data rules) |

## Commands

```sh
make install   # uv sync --extra dev + pre-commit hooks
make check     # ALL gates — must pass before every commit
make format    # ruff format + lint autofix
make hooks     # pre-commit on all files
uv run agavus  # run the app
```

Never use `--no-verify`. Never relax a config (thresholds, limits, ignores)
to make a gate pass — fix the code.

## Machine-enforced rules

1. **2 spaces**, no tabs (Makefile recipes excepted — Make requires tabs).
2. **mypy strict**: annotate every input/output; no untyped defs, no `Any`,
   no type-ignore.
3. **Function ≤ 80 code lines; file ≤ 500** (comments/blanks excluded).
   Split code; never raise limits.
4. **Zero duplication**: shared token runs ≥ 30 fail the build. Extract
   helpers (scripts share `scripts/pyutils.py`).
5. **No unused deps** (deptry). CLI-only tools need a `DEP002` entry in
   `[tool.deptry.per_rule_ignores]`.
6. **Tests in per-domain `tests/` folders**: `src/agavus/<domain>/tests/`
   (e.g. `corpus/tests/storage_test.py`); tooling tests in `scripts/tests/`.
   Naming stays `<module>_test.py`. Every new module gets tests.
7. **Line length ≤ 100**; ruff lint + format clean.

## Project invariants (violating these is a design bug)

- **Pure Python, CPU-only.** Never add Node/npm deps or GPU-required code.
  The LLM lab (`docs/llm-lab.md`) may *use* a GPU but never require one.
  Everything runs through `uv`.
- **Corpus never in git.** Public repo + LGPD. Real data only in gitignored
  `data/` (`AGAVUS_DATA_DIR`); git holds code, taxonomy tables, and
  **synthetic** fixtures only. Never weaken the staged-data guard.
- **Provenance is mandatory**: every label carries `label_source`, every
  text carries a verbatim-citable `origin_ref`. Conflicts go to review,
  never silently resolved.
- **Extensibility seams**: labels flow through the area registry — no
  hardcoded `F`/`M` literals outside pack definitions; loss matrix is
  composed data; pipelines take an area parameter.
- **Safety ceiling**: M→F loss = 10 and Reject|M = 8 are global maximums.
  No change may make Maria da Penha matters easier to reject or misroute.
- **Answers are cited excerpts, never generated opinions.** When uncertain,
  defer (three-way decision) — do not guess. No legal advice.
- **Config only in `agavus.config.settings`** (frozen typed `Settings`).
  New config = env var + field + `.env.example` entry. Never read
  `os.environ` elsewhere.
- **GPU Brasil**: approval before every create, start, stop, rename or delete;
  never print keys or `disk_token`; always attach disk `agavus-training` and
  save all lab work to it; case-derived text only as the anonymized,
  scan-filtered export allowed by `docs/llm-lab.md` §3 (`docs/gpu-brasil.md`).
- **Scope**: TJDFT, all dates the source holds (today backward until empty),
  domains F + M only
  (general criminal is out of scope). Do not widen scope without a design
  doc change first.

## Implementation style

- Domain-first packages (`src/agavus/<domain>/`); no technical-layer folders.
- Frozen dataclasses for value types; pure functions where possible;
  module + function docstrings; small, explicitly typed public surfaces.
- Ingestion: deterministic ids, idempotent, resumable; raw never mutated.
- Train/held-out splits by case (prefer time) — never leak the same case
  across splits.

## Commits — semantic and concise

`type(scope): subject` — imperative, ≤ 72 chars (aim ≤ 60), no trailing
period. Body **only when the why is not obvious**, a few short lines.
`design` commits: one-line decision + `Refs:`; safety-related changes state
"M ceiling preserved". Types, scopes, rules: `docs/commits.md` (enforced by
the commit-msg hook). One coherent change per commit; no WIP.

## Workflow

1. Read the relevant design section (table above).
2. Smallest coherent change, consistent with surrounding code.
3. Add/update tests in the domain's `tests/` folder (synthetic data only).
4. `make check` — everything green.
5. Commit with a clear message; hooks run automatically. Push only when
   asked or when the task clearly includes it.