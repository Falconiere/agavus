"""Private, read-only corpus selection with snapshot-only negative labels."""

import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass

from agavus.classify.learned.partitions import Partitions, partition
from agavus.classify.learned.scenarios import relation_samples
from agavus.classify.learned.types import Sample
from agavus.classify.pipeline import _CNJ
from agavus.classify.router import TRAINING
from agavus.corpus.anonymize import contains_personal_data
from agavus.corpus.db import _row_dict
from agavus.corpus.records import LabelSource, TextDocument, TextSource, text_from_dict
from agavus.corpus.storage import CorpusStorage
from agavus.taxonomy.areas import AREAS, OUT_OF_SCOPE
from agavus.taxonomy.labeling import LabelDecision, label_text
from agavus.taxonomy.organs import ORGAN_RULES_VERSION


@dataclass(frozen=True)
class Dataset:
  """Frozen examples, membership and data-quality counts; no corpus writes."""

  samples: tuple[Sample, ...]
  partitions: Partitions
  report: dict[str, object]


def _documents(storage: CorpusStorage) -> Iterator[TextDocument]:
  """Stream the cache without loading 100,000 full decisions into memory."""
  path = (storage.processed / "corpus.db").resolve()
  with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as conn:
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM texts ORDER BY id")
    for row in cursor:
      yield text_from_dict(_row_dict(row, ("citations",)))


def _sample(document: TextDocument, decision: LabelDecision, *, derived: bool = False) -> Sample:
  match = _CNJ.search(document.origin_ref)
  case = re.sub(r"\D", "", match[0]) if match else document.origin_ref
  return Sample(
    document.id,
    document.text,
    decision.domain_label or "",
    case,
    document.origin_ref,
    decision.label_source.value + (f":snapshot-derived:{ORGAN_RULES_VERSION}" if derived else ""),
    document.published_at,
    document.source == TextSource.STATUTE,
  )


def _collect(storage: CorpusStorage) -> tuple[list[Sample], dict[str, Sample], dict[str, int]]:
  decisions = {d.record_id: d for d in storage.load_labels()}
  blocked = _blocked_cases(storage, decisions)
  existing_cases: set[str] = set()
  selected: list[Sample] = []
  candidates: dict[str, Sample] = {}
  counts = dict.fromkeys(("review", "privacy", "missing_case", "unlabeled"), 0)
  for document in _documents(storage):
    resolved = decisions.get(document.id)
    match = _CNJ.search(document.origin_ref)
    case = re.sub(r"\D", "", match[0]) if match else ""
    if case in blocked or (resolved is not None and resolved.needs_review):
      counts["review"] += 1
      continue
    derived = resolved is None
    decision = resolved or label_text(document)
    if decision.needs_review or decision.domain_label not in AREAS:
      counts["unlabeled"] += 1
      continue
    if derived and decision.domain_label != OUT_OF_SCOPE.code:
      continue
    if document.source != TextSource.STATUTE and _CNJ.search(document.origin_ref) is None:
      counts["missing_case"] += 1
      continue
    if contains_personal_data(document.text) or contains_personal_data(document.origin_ref):
      counts["privacy"] += 1
      continue
    sample = _sample(document, decision, derived=derived)
    if derived:
      candidates.setdefault(sample.group, sample)
    else:
      selected.append(sample)
      existing_cases.add(sample.group)
  return selected, {g: s for g, s in candidates.items() if g not in existing_cases}, counts


def _blocked_cases(storage: CorpusStorage, decisions: dict[str, LabelDecision]) -> set[str]:
  """A review flag cannot be bypassed by choosing a sibling text from the case."""
  review_ids = {
    key
    for key, decision in decisions.items()
    if decision.needs_review or decision.label_source == LabelSource.PENDING
  }
  references = [s.origin_ref for s in storage.load_texts_by_ids(tuple(review_ids)).values()]
  references.extend(case.id for case in storage.load_cases() if case.id in review_ids)
  return {re.sub(r"\D", "", match[0]) for ref in references if (match := _CNJ.search(ref))}


def prepare_dataset(storage: CorpusStorage, negative_cases: int = 100) -> Dataset:
  """Select distinct negative cases deterministically; fail rather than weaken splits."""
  if negative_cases < 0:
    raise ValueError("negative case count must be nonnegative")
  selected, candidates, excluded = _collect(storage)
  if len(candidates) < negative_cases:
    raise ValueError(
      f"insufficient eligible negative cases: {len(candidates)} < {negative_cases}; "
      f"exclusions={excluded}"
    )
  selected.extend(candidates[group] for group in sorted(candidates)[:negative_cases])
  case_labels = {sample.label for sample in selected if not sample.auxiliary}
  if case_labels != set(AREAS):
    raise ValueError(f"missing resolved case classes: {sorted(set(AREAS) - case_labels)}")
  selected.extend(
    Sample(
      f"query:{i}",
      text,
      label,
      f"query:{i}",
      f"authored-query:{i}",
      "synthetic-authored",
      auxiliary=True,
    )
    for i, (text, label) in enumerate(TRAINING)
  )
  samples = tuple(selected) + relation_samples()
  parts = partition(samples)
  report: dict[str, object] = {
    "excluded": excluded,
    "negative_candidates": len(candidates),
    "negative_cases_added": negative_cases,
    "partition_counts": {
      name: len(getattr(parts, name)) for name in ("train", "validation", "calibration", "final")
    },
    "classes": {
      label: sum(s.label == label and not s.auxiliary for s in samples) for label in AREAS
    },
    "query_count": len(TRAINING),
    "relation_pairs": len(relation_samples()),
  }
  return Dataset(samples, parts, report)
