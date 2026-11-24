# Schema — what a record looks like

Two frozen, typed record types (future module: `corpus/records`). Every
record carries **provenance**: where it came from and where its label came
from. Synthetic examples only — real data never enters git.

## CaseRecord (metadata, from DataJud)

| Field | Type | Meaning |
| ------------------ | -------------- | -------------------------------------------- |
| `id` | str | process number (CNJ format) |
| `tribunal` | str | always `TJDFT` in this phase |
| `orgao_julgador` | str | vara/turma name — weak-label signal |
| `classe_tpu` | str | TPU process class code |
| `assuntos_tpu` | list[str] | TPU subject codes |
| `data_ajuizamento` | date | filing date (window filter) |
| `movimentos` | list[Movement] | dated procedural events |
| `domain_label` | F \| M \| O | routing label |
| `label_source` | enum | `tpu` \| `organ` \| `manual` \| `weak` |
| `raw_ref` | str | pointer to the raw payload |

## TextDocument (full text, from DJEN/portal/statutes)

| Field | Type | Meaning |
| -------------- | ----------- | --------------------------------------------- |
| `id` | str | stable hash of source + origin_ref |
| `source` | enum | `djen` \| `portal` \| `statute` |
| `origin_ref` | str | canonical public reference (verbatim citable) |
| `published_at` | date | publication date (window filter) |
| `text` | str | anonymized, normalized full text |
| `citations` | list[str] | regex-extracted ("art. 22, III, Lei 11.340/2006") |
| `domain_label` | F \| M \| O | routing label |
| `label_source` | enum | same provenance enum |
| `raw_ref` | str | pointer to the raw payload |

## Synthetic example

```
CaseRecord:
  id: "0701234-56.2024.8.07.0001"
  orgao_julgador: "1ª Vara de Violência Doméstica e Familiar"
  assuntos_tpu: ["Violência Doméstica", "Medida Protetiva"]
  domain_label: M
  label_source: tpu

TextDocument:
  source: djen
  origin_ref: "Acórdão 1234567, 2ª Turma Criminal, DJEN 2024-05-10"
  text: "...DEFIRO a medida protetiva de urgência... art. 22, III, Lei 11.340/2006..."
  citations: ["art. 22, III, Lei 11.340/2006"]
  domain_label: M
  label_source: organ
```

## Invariants

1. `label_source` is **mandatory** — weak labels stay auditable; conflicts
   go to review, never silently resolved ([labeling.md](labeling.md)).
2. `origin_ref` must be **verbatim citable** — answers quote it as-is.
3. `id` is deterministic → re-ingestion deduplicates naturally.
4. No party names, documents (CPF/RG), or addresses anywhere
   ([privacy.md](privacy.md)).
5. Records are frozen: processing produces new versions, never edits.