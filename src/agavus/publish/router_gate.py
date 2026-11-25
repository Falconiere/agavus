"""Release gate for the question router the Space serves."""

from agavus.classify.router_store import ensure_router
from agavus.config.settings import Settings
from agavus.evals.goldset import ROUTING_V1
from agavus.evals.routing import RoutingReport, evaluate_routing
from agavus.evals.routingset import DEV
from agavus.evals.run import EvalReport, evaluate
from agavus.publish.bootstrap import model_path
from agavus.publish.export import gate
from agavus.retrieve.dense import prepare_encoder

MIN_DEV_M_RECALL = 0.90


def router_problems(gold: EvalReport, dev: RoutingReport) -> list[str]:
  """The gold gate, plus no unsafe decision and high M recall on DEV."""
  problems = gate(gold)
  if dev.unsafe:
    problems.append(f"M questions misrouted or rejected: {', '.join(dev.unsafe)}")
  if dev.m_hits < MIN_DEV_M_RECALL * dev.m_total:
    problems.append(f"DEV M-recall {dev.m_hits}/{dev.m_total} < {MIN_DEV_M_RECALL}")
  return problems


def ensure_router_passes(settings: Settings) -> RoutingReport:
  """Raise unless the router is available and passes; an ungated one never ships."""
  encoder = prepare_encoder(settings.data_dir)
  if encoder is None:
    raise ValueError("encoder unavailable: the Space router cannot be gated")
  router = ensure_router(model_path(settings), encoder)
  gold = evaluate(router.probabilities([item.text for item in ROUTING_V1]), ROUTING_V1)
  dev = evaluate_routing(DEV, router.probabilities([item.text for item in DEV]))
  problems = router_problems(gold, dev)
  if problems:
    raise ValueError("router gate failed: " + "; ".join(problems))
  return dev
