# Commit conventions — semantic and concise

Commits are the running memory of the project: **semantic** (structured,
machine-readable) and **concise** (say it in one line when one line is
enough). Enforced by the `commit-msg` hook (`scripts/check_commit_msg.py`).

## Format

```
type(scope): imperative subject — max 72 chars, aim ≤ 60, no trailing period

Optional body: only when the WHY is not obvious from the subject.
A few short lines, wrapped at 100. No essays, no restating the diff.

Refs: docs/DESIGN.md §7        ← only for design commits
```

## Types

| Type | Meaning |
| ---------- | ---------------------------------------------- |
| `feat` | new capability in a domain |
| `fix` | bug fix |
| `design` | spec/decision change (scope, loss matrix, data range, architecture) |
| `docs` | explanatory docs, no spec change |
| `refactor` | behavior-preserving code change |
| `test` | tests only |
| `chore` | tooling, config, deps, hooks |
| `build` | build/packaging/CI |

`design` = changes what/how we build. `docs` = only explains.

## Scopes (closed list)

Domains: `corpus`, `taxonomy`, `classify`, `decide`, `retrieve`, `answer`,
`config`, `greeting`. Cross-cutting: `docs`, `scripts`, `tooling`, `repo`.

One scope per commit; span two domains → split into two commits.

## Rules

1. **Subject = what, body = why — and only if needed.** Most commits need
   no body. The diff shows what; add a body only for non-obvious decisions.
2. **`design` commits**: one-line decision + `Refs:` to the doc/section.
   Record rejected alternatives only when future agents might re-litigate.
3. **Safety line** when touching loss matrix/thresholds/M routing: state
   "M ceiling preserved" (or that a design change deliberately alters it).
4. **One coherent change per commit.** No WIP, no "misc fixes"; every
   commit revertible as a unit.
5. Imperative mood: "add", "narrow", "enforce" — not "added"/"adds".

## Examples

Good — concise, no body needed:

```
feat(corpus): add DataJud TJDFT ingestion
fix(decide): correct beta threshold for M
test(classify): cover negation-scoped features
```

Good — design decision, short body:

```
design(decide): narrow scope to F+M

M absorbs all DV; 3-domain split rejected (M↔P overlap, no safety gain).
M ceiling preserved. Refs: docs/DESIGN.md §2, §7
```

Bad: `updated stuff`, `fix`, `WIP`, `feat: changes`, or a 30-line body
restating the diff.

## Enforcement

- Hook validates: type, scope, header ≤ 72, no trailing period, blank line
  before body, body lines ≤ 100. Conciseness of spirit is on us.
- `Merge …` / `Revert …` pass through. History before this convention is
  grandfathered.