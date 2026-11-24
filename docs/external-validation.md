# External validation — Jev as comparator, not dependency

Status: **pilot executed 2026-11-24, results below**. TypeSafe's Jev (a
"System One model": unstructured state in → typed, calibrated decisions out)
validates our architecture — agavus's `classify → decide` contract is the
same shape, domain-specialized and local. Jev's role here is **external
validator/comparator on synthetic data**, never a production dependency.

## Tooling record

- Skills: `wuyoscar/jev-skill` five-entry preview @ commit `7ca55e1` —
  `jev`, `jev-triage`, `jev-documents`, `jev-eval` installed user-wide
  (`~/.config/opencode/skills/`); `jev-act` skipped (irrelevant).
- Runtime: `jev-decide` v0.2.1 (uv tool, stdlib-only). Provider: official
  TypeSafe API only (repo policy rejects relays; matches our privacy stance).
- Credentials: `TYPESAFE_API_KEY` in the user's interactive fish config;
  presence checked, values never printed or stored in the repo.

## Skill → validation mapping

| Skill | Validates | agavus counterpart |
| --------------- | -------------------------------------------- | ---------------------------- |
| `jev-triage` | F/M/O routing labels on gold records | `classify` + `decide` |
| `jev-documents` | claim-to-source support (citation correctness, groundedness) | response-quality checks 2–3 |
| `jev-eval` | rubric judgments on answers | eval harness, human rubric |

## Privacy rules (non-negotiable)

1. Only **authored/synthetic records and public-domain statute text** go to
   the API. Never real TJDFT case data, even anonymized (LGPD).
2. Official TypeSafe API only; no relays/proxies.
3. Paid calls require explicit approval; pilots are bounded and receipted.
4. Gold labels are frozen before predictions and never sent to any model.

## Pilot protocol (executed)

10 synthetic pt-BR records, stratified: clear-F ×2, clear-M ×2, near-miss
M↔F ×2 (alienação parental; mixed threats+pensão with safety priority),
OOD ×2 (criminal, trabalhista), vague ×1, calm psychological/patrimonial
violence ×1. Gold frozen first; one batched request, one Choice per record
ID; shared categories + safety-priority policy; dry-run before live call.

## Results

**Triage — 10/10 vs synthetic gold:**

| Record | Stratum | Gold | Jev | P |
| ------ | ------------------------------ | ---- | ---- | ----- |
| r01 | clear-F (divórcio/partilha) | F | F | 1.00 |
| r02 | clear-F (alimentos) | F | F | 1.00 |
| r03 | clear-M (ameaça + protetiva) | M | M | 1.00 |
| r04 | clear-M (descumprimento) | M | M | 1.00 |
| r05 | near-miss (alienação parental) | F | F | 0.99 |
| r06 | near-miss (ameaças + pensão) | M | M | 1.00 |
| r07 | OOD (tráfico) | O | O | 1.00 |
| r08 | OOD (rescisão trabalhista) | O | O | 1.00 |
| r09 | vague ("não deixa ver meu filho") | F | F | 0.98 |
| r10 | M calma (psicológica/patrimonial) | M | M | 1.00 |

**Documents — claim-to-source (Lei 12.318/2010, art. 2º excerpt):**
true claim → `true` (p 0.97) ✅; hallucinated penalty claim → `false`
(p 0.02) ✅. The exact verifier behavior our response-quality contract needs.

**Eval — rubric scores:** well-cited answer → 1.99/2 ✅; wrong-article +
advice-language answer → 0.0/2 ✅.

**Receipts:** model `jev-1.13.0` (requested `typesafe/jev-1.13`);
latency 0.10–0.15s/call; tokens: triage 2059 in/383 out, documents 570/42,
eval 775/32; cost ≈ $0.0001 input at published pricing (output free).
`jev_called: true`, mode `jev_api`, transport `typesafe`.

## Second call: taxonomy policy validation (2026-11-24)

One batched request, four questions, synthetic policy scenarios only:

| Question | Jev judgment | P | Adopted in code |
| ------------------------------ | ------------------ | ----- | ------------------------------ |
| Júri without TPU subjects | review | 0.98 | `organs.py`: júri → None (review) |
| Mixed M+F subjects | M provisional + review | 0.99 | `labeling.py`: safety priority |
| "Vara de Família e Sucessões" → F | true | 0.78 | rule kept, **weak** source (below 0.8 review threshold — treated accordingly) |
| Lei 12.318/2010 → F | true | 0.94 | `STATUTE_AREA` map |

Note the CLI's own review policy fired (exit 2) on the 0.78 judgment —
the tooling practices what we preach: confidence below threshold → review.

## Caveats (read before trusting)

1. **Tiny synthetic pilot ≠ production accuracy or calibration.** Near-1.0
   probabilities on clean records say nothing about hard real-world cases.
2. Gold is authored by us — agreement with our expectations, not independent
   truth. Scale needs a larger, reviewed gold set.
3. Jev classifies confidently where agavus may **defer** (r09: p 0.98).
   That is the design difference: Jev supplies judgments; **decisions live
   in our code** (loss matrix, safety ceiling). Its confidence is an input,
   never an authority.
4. Provider evals elsewhere are English/business-weighted; pt-BR legal
   looked strong here but needs the full gold-set benchmark.

## Planned use (each gated, each approval-bounded)

1. `jev-triage` as **routing comparator** on the gold set when `classify`
   lands: disagreements → review queue, never auto-correction.
2. `jev-documents` as **citation auditor** in the eval harness (sampled
   accepted answers, synthetic/public sources only).
3. `jev-eval` as **external rubric judge** beside human evaluation.
4. Optional: frontier-reference probabilities for workflow evals (their
   evals philosophy), always on synthetic data.

Scaling any of these requires: larger reviewed gold set, explicit budget
approval, and the regression policy from [evals.md](evals.md).