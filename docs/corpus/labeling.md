# Labeling — where F/M/O comes from

Three label sources, in priority order. Every label records its source
(`label_source`) — provenance is mandatory, not optional.

## 1. TPU codes (primary, `label_source: tpu`)

DataJud records carry CNJ TPU classe/assunto codes. A curated, versioned
mapping table converts codes → F/M/O:

| TPU signal (illustrative) | Label |
| -------------------------------------------------- | ----- |
| violência doméstica / medida protetiva | M |
| feminicídio / lesão corporal em contexto DV | M |
| divórcio, guarda, alimentos, união estável | F |
| general criminal (tráfico, roubo, estelionato…) | O |
| anything else | O |

The table lives in git (small, public taxonomy — not case data), in the
future `taxonomy/tpu` module. Unknown codes → O or review, never guessed.

## 2. Órgão julgador weak labels (`label_source: organ` | `weak`)

TJDFT specialized organs are strong signals for full texts without TPU codes:

| Organ pattern | Label |
| -------------------------------------------------------- | ----- |
| Vara de Violência Doméstica e Familiar contra a Mulher | M |
| Vara de Família (incl. "e Sucessões") | F |
| Vara Criminal | O (general criminal is out of scope) |
| Tribunal do Júri | **review** until TPU subjects confirm feminicídio/DV → M (Jev-validated: never guess between M and O) |

Rules are versioned and auditable. A document from a violência doméstica
vara is *almost certainly* M — "almost" is why these are **weak** labels.

**Implemented policy** (`taxonomy/labeling.py`): TPU mapping is primary;
organ rules are the weak fallback. Multiple TPU areas (e.g. Violência
Doméstica + Guarda) → **provisional M + review** (safety priority,
Jev-validated 2026-11-24). TPU-vs-organ conflicts keep the TPU label but
are flagged for review. The TPU code table is seeded only with verified
codes (unknown codes never guess); it grows as real DataJud records arrive.

## 3. Manual gold set (`label_source: manual`)

Small curated set of queries + documents, human-labeled. The only fully
trusted labels; used for calibration and evaluation. Grows over time:
reviewed boundary cases flow back in (active learning, most-uncertain first).

## Conflict policy

- TPU says F, organ says M (real overlap: protective measures with custody)?
  **Never silently resolved.** Record lands with the higher-priority source,
  the conflict is logged, and the case enters the review queue.
- These conflicts are exactly the M↔F overlap the design expects — they are
  data for the boundary region, not noise to delete.

## Audit trail

For every record: `label_source` + mapping/rule version + (if reviewed)
reviewer decision. Any label can be traced back to *why* it exists.