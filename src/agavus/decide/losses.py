"""Cost-sensitive loss matrix, composed from the area registry.

Pack data (docs/DESIGN.md §7): safety ceiling M→F = 10 and Reject|M = 8 are
global maximums — no area pack may make Maria da Penha easier to reject or
misroute. Units: 1 = cost of one deferral round.
"""

from agavus.taxonomy.areas import AREAS, OUT_OF_SCOPE, routing_areas

DEFER_COST = 1.0
ACCEPT_OUT_OF_SCOPE_COST = 8.0
DEFAULT_REJECT_COST = 4.0
DEFAULT_CROSS_COST = 4.0
# (true area, accepted-as area) -> cost; asymmetric by design.
CROSS_COSTS: dict[tuple[str, str], float] = {("M", "F"): 10.0, ("F", "M"): 6.0}
REJECT_COSTS: dict[str, float] = {"M": 8.0}
SAFETY_CEILING = 10.0


def accept_action(area_code: str) -> str:
  """Canonical action name for accepting an area."""
  return f"accept-{area_code}"


def build_matrix() -> dict[str, dict[str, float]]:
  """Compose the full loss matrix from registry areas + pack costs."""
  states = tuple(AREAS)
  matrix: dict[str, dict[str, float]] = {}
  for area in routing_areas():
    row: dict[str, float] = {}
    for state in states:
      if state == area.code:
        row[state] = 0.0
      elif state == OUT_OF_SCOPE.code:
        row[state] = ACCEPT_OUT_OF_SCOPE_COST
      else:
        row[state] = CROSS_COSTS.get((state, area.code), DEFAULT_CROSS_COST)
    matrix[accept_action(area.code)] = row
  matrix["reject"] = {
    state: (0.0 if state == OUT_OF_SCOPE.code else REJECT_COSTS.get(state, DEFAULT_REJECT_COST))
    for state in states
  }
  matrix["defer"] = dict.fromkeys(states, DEFER_COST)
  return matrix


def validate_safety(matrix: dict[str, dict[str, float]]) -> None:
  """Enforce the safety ceiling; raises on any violation."""
  for action, row in matrix.items():
    for state, cost in row.items():
      if cost > SAFETY_CEILING:
        raise ValueError(f"loss {action}|{state} = {cost} exceeds ceiling {SAFETY_CEILING}")
  m_code = next(a.code for a in routing_areas() if a.code == "M")
  if matrix[accept_action("F")][m_code] != SAFETY_CEILING:
    raise ValueError("M→F misroute cost must equal the safety ceiling")
  if matrix["reject"][m_code] != REJECT_COSTS[m_code]:
    raise ValueError("Reject|M cost must equal the safety pack value")
