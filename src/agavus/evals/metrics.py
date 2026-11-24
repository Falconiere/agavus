"""Classification metrics: accuracy, macro-F1, confusion (docs/evals.md)."""

from collections.abc import Sequence


def accuracy(predictions: Sequence[str], gold: Sequence[str]) -> float:
  """Fraction of exact matches."""
  if not gold:
    return 0.0
  hits = sum(pred == truth for pred, truth in zip(predictions, gold, strict=True))
  return hits / len(gold)


def confusion(
  predictions: Sequence[str], gold: Sequence[str], classes: Sequence[str]
) -> dict[str, dict[str, int]]:
  """counts[true][predicted], zero-filled over classes."""
  matrix = {truth: dict.fromkeys(classes, 0) for truth in classes}
  for pred, truth in zip(predictions, gold, strict=True):
    if truth in matrix and pred in matrix[truth]:
      matrix[truth][pred] += 1
  return matrix


def macro_f1(predictions: Sequence[str], gold: Sequence[str], classes: Sequence[str]) -> float:
  """Unweighted mean of per-class F1."""
  matrix = confusion(predictions, gold, classes)
  scores: list[float] = []
  for cls in classes:
    tp = matrix[cls][cls]
    fp = sum(matrix[other][cls] for other in classes if other != cls)
    fn = sum(matrix[cls][other] for other in classes if other != cls)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
  return sum(scores) / len(scores) if scores else 0.0
