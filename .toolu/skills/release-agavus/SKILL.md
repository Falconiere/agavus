---
name: release-agavus
description: Release Agavus through signed tags, gated Actions, and public HF publication.
metadata:
  toolu:
    origin: agent
    created: 2026-11-26T19:53:48Z
---
## When to Use

Use when the user asks to publish a new Agavus version. Follow `docs/publishing.md`.

## Procedure

1. Inspect status, current package version, tags, releases and commits since the last tag. Preserve
  unrelated work. Choose the next semantic version from actual changes.
2. Bump `pyproject.toml`, run `uv lock`, `uv build` and `make check`. Inspect wheel/source contents
  to exclude private data, credentials and experiment artifacts.
3. Preflight in a fresh ignored data directory. Bootstrap only `space/assets`, assert no
  cases/non-statute texts, and run `stage_release`, `ensure_router_passes` and `stage_space` without
  uploading. Use `agavus.publish.export`, `.router_gate` and `.space_release`; pass the new tag
  to Space staging.
4. Commit the version bump with an Intent trailer; push `HEAD:main` and wait for CI. Preserve
  configured signing: create and verify a signed version tag matching the package version. Push only
  that tag to start the Release workflow.
5. Watch Release through package building, gated model/Space publication and GitHub Release assets.
  Resolve failures without weakening gates; do not claim success from tag creation alone.
6. Verify all five release assets, compare the HF model checksum with the GitHub model asset, and
  check the Space requirements pin. Wait for `runtime.stage=RUNNING` and `runtime.sha` matching the
  Space repository commit; check the live API against its source contract.

## Pitfalls

- The normal `data/` contains private cases. Never run publication against it; use a fresh
  public-only namespace.
- Private learned checkpoints remain local and inactive, regardless of releasing their optional
  source code.
- Do not force-update published tags, discard signatures, publish case-derived data, or change
  safety thresholds.
- A successful upload can precede Space startup. Its repository SHA alone does not prove the running
  app updated.
- GitHub's unsupported push-certificate warning does not invalidate commit/tag signatures.

## Verification

- Package version, verified tag and GitHub Release agree; main is pushed and the worktree is clean.
- Full checks and Release workflow succeed; wheel, source, model card, eval report and public model
  database are uploaded.
- HF model bytes match the release asset; the Space pins the tag and serves the matching commit.
- Report source/package publication separately from activation of experimental models.
