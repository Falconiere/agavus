"""Three-way decision rule: minimize expected loss (Yao / DTRS).

Accept / reject / defer — probability alone never decides; cost does.
Ties prefer defer (the safe harbor). Thresholds alpha/beta are the rule pre-solved
(docs/math/three-way-decisions.md).
"""

from collections.abc import Mapping
from dataclasses import dataclass

from agavus.decide.losses import accept_action, build_matrix, validate_safety
from agavus.taxonomy.areas import OUT_OF_SCOPE, routing_areas


@dataclass(frozen=True)
class Decision:
  """The chosen action plus the full expected-loss table (auditability)."""

  action: str
  expected_losses: dict[str, float]


def expected_losses(
  probabilities: Mapping[str, float], matrix: dict[str, dict[str, float]]
) -> dict[str, float]:
  """R(action|x) = sum of loss(action|state) * P(state|x)."""
  return {
    action: sum(cost * probabilities.get(state, 0.0) for state, cost in row.items())
    for action, row in matrix.items()
  }


def decide(probabilities: Mapping[str, float]) -> Decision:
  """Minimum-expected-loss action; ties prefer defer, then stable order."""
  matrix = build_matrix()
  validate_safety(matrix)
  losses = expected_losses(probabilities, matrix)
  priority = {"defer": 0, "reject": 1}
  action = min(losses, key=lambda name: (losses[name], priority.get(name, 2), name))
  return Decision(action=action, expected_losses=losses)


def thresholds(area_code: str) -> tuple[float, float]:
  """Accept (alpha) and reject (beta) thresholds from the binary reduction."""
  matrix = build_matrix()
  accept = matrix[accept_action(area_code)]
  states = tuple(matrix["defer"])
  lambda_pp = accept[area_code]
  lambda_pn = sum(accept[s] for s in states if s != area_code) / (len(states) - 1)
  lambda_bp = matrix["defer"][area_code]
  lambda_bn = matrix["defer"][OUT_OF_SCOPE.code]
  lambda_np = matrix["reject"][area_code]
  lambda_nn = matrix["reject"][OUT_OF_SCOPE.code]
  alpha = (lambda_pn - lambda_bn) / ((lambda_pn - lambda_bn) + (lambda_bp - lambda_pp))
  beta = (lambda_bn - lambda_nn) / ((lambda_bn - lambda_nn) + (lambda_np - lambda_bp))
  return alpha, beta


def routing_codes() -> tuple[str, ...]:
  """Areas that can be accepted (registry-driven)."""
  return tuple(area.code for area in routing_areas())
