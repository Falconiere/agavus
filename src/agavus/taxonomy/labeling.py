"""Labeling pass: TPU first, organ second; conflicts to review, never silent.

Mixed M+F signals yield provisional M + review (safety priority,
Jev-validated). Statutes map via a curated table (manual source).
"""

from dataclasses import dataclass

from agavus.corpus.records import CaseRecord, LabelSource, TextDocument
from agavus.taxonomy import organs, tpu
from agavus.taxonomy.areas import FAMILIA, MARIA_DA_PENHA

STATUTE_AREA: dict[str, str] = {
  "Lei 11.340/2006": MARIA_DA_PENHA.code,
  "Lei 14.994/2024": MARIA_DA_PENHA.code,
  "Lei 12.318/2010": FAMILIA.code,
  "Lei 13.058/2014": FAMILIA.code,
  "Lei 10.406/2002": FAMILIA.code,
}


@dataclass(frozen=True)
class LabelDecision:
  """Auditable labeling outcome for one record."""

  record_id: str
  domain_label: str | None
  label_source: LabelSource
  needs_review: bool
  reason: str


def label_case(record: CaseRecord) -> LabelDecision:
  """Label a case: TPU primary, organ weak fallback, conflicts flagged."""
  tpu_areas = tpu.map_codes(record.assuntos_tpu)
  organ = organs.organ_label(record.orgao_julgador)
  if len(tpu_areas) > 1:
    provisional = MARIA_DA_PENHA.code if MARIA_DA_PENHA.code in tpu_areas else None
    return LabelDecision(record.id, provisional, LabelSource.WEAK, True, "multiple TPU areas")
  if tpu_areas:
    area = next(iter(tpu_areas))
    conflict = organ is not None and organ != area
    reason = "tpu vs organ conflict" if conflict else "tpu mapping"
    return LabelDecision(record.id, area, LabelSource.TPU, conflict, reason)
  if organ is not None:
    return LabelDecision(record.id, organ, LabelSource.ORGAN, False, "organ weak label")
  return LabelDecision(record.id, None, LabelSource.PENDING, True, "no signal")


def label_text(document: TextDocument) -> LabelDecision:
  """Label a text: statute map, then organ name carried in origin_ref."""
  for name, area in STATUTE_AREA.items():
    if document.origin_ref.startswith(name):
      return LabelDecision(document.id, area, LabelSource.MANUAL, False, f"statute map: {name}")
  organ = organs.organ_label(document.origin_ref)
  if organ is not None:
    return LabelDecision(document.id, organ, LabelSource.ORGAN, False, "organ in origin_ref")
  if "tribunal do júri" in document.origin_ref.casefold():
    return LabelDecision(document.id, None, LabelSource.PENDING, True, "júri without tpu")
  return LabelDecision(document.id, None, LabelSource.PENDING, True, "no signal")
