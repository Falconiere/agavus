"""A confidently rejected M case must be visible as an unsafe decision."""

import numpy as np

from agavus.classify.learned.metrics import area_metrics, relation_metrics
from agavus.classify.learned.types import RELATIONS, Sample
from agavus.taxonomy.areas import FAMILIA, MARIA_DA_PENHA, OUT_OF_SCOPE


def test_metrics_expose_unsafe_rejection_and_relation_category_failures() -> None:
  classes = (FAMILIA.code, MARIA_DA_PENHA.code, OUT_OF_SCOPE.code)
  sample = Sample("case", "fictional", MARIA_DA_PENHA.code, "case", "synthetic:1", "manual")
  scores = area_metrics([sample], np.array([[0.0, 0.0, 20.0]]), classes, 1.0)
  assert scores["unsafe_m_decisions"] == ["case"]
  assert scores["m_recall"] == 0.0
  relation = Sample(
    "pair",
    "fictional",
    "support",
    "family",
    "synthetic:2",
    "authored",
    statement="claim",
    category="quotation",
  )
  result = relation_metrics([relation], np.array([[0.0, 20.0, 0.0]]), RELATIONS, 1.0)
  assert result["category_failures"] == {"quotation": ["pair"]}
