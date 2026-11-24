# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project

`agavus` — Python >= 3.12, managed with `uv`. Pure Python tooling: **never add
Node/npm dependencies**. Groundwork for a Brazilian legal LLM project.

## Commands

| Task                        | Command          |
| --------------------------- | ---------------- |
| Install env + git hooks     | `make install`   |
| Run ALL gates (must pass)   | `make check`     |
| Auto-format + lint autofix  | `make format`    |
| Run hooks on all files      | `make hooks`     |
| Run the app                 | `uv run agavus`  |

Always run `make check` before committing. Pre-commit hooks enforce the same
gates on every commit; never bypass them (`--no-verify` is forbidden).

## Hard rules (machine-enforced — do not relax configs to pass)

1. **Indentation: 2 spaces**, no tabs (Makefile recipes are the only exception).
2. **Strict types**: mypy `strict = true`. Annotate every function input and
   output. No untyped defs, no `Any`, no ignored type errors.
3. **Function <= 80 code lines; file <= 500 code lines** (comments and blank
   lines excluded). Split code instead of raising limits.
4. **Zero duplication**: any shared run of >= 30 tokens fails the build
   (within or across files). Extract shared helpers instead of copying.
5. **No unused dependencies** (deptry). Every dependency must be imported;
   CLI-only tools need an explicit `DEP002` entry in `[tool.deptry.per_rule_ignores]`.
6. **Tests co-located**: `foo.py` is tested by `foo_test.py` in the same
   directory. Every new module gets tests.
7. **Line length <= 100** (ruff).

## Architecture

- Domain-first layout: `src/agavus/<domain>/` (e.g. `config/`, `greeting/`).
  New features become new domain packages, not technical layers.
- Configuration is isolated in `agavus.config.settings`: frozen, typed
  `Settings` dataclass. New config = new env var + field on `Settings` +
  entry in `.env.example`. Never read `os.environ` outside this module.
- Repo quality tooling lives in `scripts/` and shares helpers via
  `scripts/pyutils.py` — extend it rather than duplicating logic.

## Style

- Module and function docstrings; concise, imperative comments only when needed.
- Prefer frozen dataclasses for value types; pure functions where possible.
- Keep public surfaces small and explicitly typed.

## Workflow

1. Make the smallest coherent change, consistent with surrounding code.
2. Add/update co-located tests.
3. `make check` — everything must pass.
4. Commit with a clear message; hooks run automatically.