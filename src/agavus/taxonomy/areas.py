"""Area registry — the extensibility seam (docs/extensibility.md).

Labels flow through this registry; no other module hardcodes area codes.
Packs are data: adding an area means adding an AreaSpec, not engine changes.
"""

from dataclasses import dataclass
from enum import Enum


class AreaStatus(Enum):
  """Activation lifecycle of an area."""

  PLANNED = "planned"
  COLLECTING = "collecting"
  SHADOW = "shadow"
  ACTIVE = "active"


@dataclass(frozen=True)
class AreaSpec:
  """Versioned specification of one routing area (pack skeleton)."""

  code: str
  name: str
  status: AreaStatus
  tribunals: tuple[str, ...]


FAMILIA = AreaSpec(
  code="F",
  name="Direito de família",
  status=AreaStatus.COLLECTING,
  tribunals=("TJDFT",),
)
MARIA_DA_PENHA = AreaSpec(
  code="M",
  name="Maria da Penha / violência doméstica",
  status=AreaStatus.COLLECTING,
  tribunals=("TJDFT",),
)
OUT_OF_SCOPE = AreaSpec(
  code="O",
  name="Fora do escopo",
  status=AreaStatus.ACTIVE,
  tribunals=(),
)

AREAS: dict[str, AreaSpec] = {spec.code: spec for spec in (FAMILIA, MARIA_DA_PENHA, OUT_OF_SCOPE)}


def is_area(code: str) -> bool:
  """Whether a code is a registered area."""
  return code in AREAS


def routing_areas() -> tuple[AreaSpec, ...]:
  """Areas that accept routing (everything except out-of-scope)."""
  return tuple(spec for spec in AREAS.values() if spec is not OUT_OF_SCOPE)
