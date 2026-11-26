# Area labels and provenance

Run `agavus label` after ingestion. It writes derived labels and review reasons in SQLite
while preserving the original records. Every label has a source, including `pending`
when the area is unknown. [Label rules](../../src/agavus/taxonomy/labeling.py)

| Input | Rule |
| --- | --- |
| DataJud case | Try verified TPU **subject** codes first. The shipped mapping table is empty. If no code resolves, use the issuing court's organ rule or flag the record for review. |
| Multiple TPU areas | Mark for review. When M is present, retain it provisionally. |
| TPU and organ disagree | Retain the TPU result and flag the conflict for review. |
| Statute text | Assign the curated area from the statute name in the public reference, with manual provenance. |
| Other text | Match an organ in the reference. If unresolved, keep the label pending and mark for review. |

A specialized domestic-violence court can supply a weak M label. A jury court alone
cannot distinguish a domestic-violence case from a general criminal case. Classifier
training excludes unresolved and review-flagged labels. The authored routing gold set
is maintained separately. [TPU table](../../src/agavus/taxonomy/tpu.py) ·
[Organ rules](../../src/agavus/taxonomy/organs.py) · [Evaluation](../evals.md)
