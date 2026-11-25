"""Answer composition: extractive, cited, honest (docs/response-quality.md).

Templates only — no generation, so no advice language and no ungrounded
claims by construction. Pre-delivery verifier enforces non-empty sources,
disclaimer presence, zero personal data, and advice-language lint before
any accepted response ships.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from agavus.corpus.anonymize import anonymize, contains_personal_data
from agavus.corpus.statutes import TOPICLESS_NAMES, extract_citations
from agavus.corpus.storage import CorpusStorage
from agavus.decide.rule import decide
from agavus.reason.chain import Chain, build_chain
from agavus.retrieve.dense import search_statutes
from agavus.retrieve.encoder import Encoder
from agavus.retrieve.search import Hit, search
from agavus.taxonomy.areas import AREAS, OUT_OF_SCOPE
from agavus.taxonomy.lexicon import QUERY_FRAMING, content_tokens

DISCLAIMER = "Isto não é aconselhamento jurídico; consulte um advogado."
SCOPE_MESSAGE = (
  "Fora do escopo: agavus cobre direito de família e violência doméstica (Maria da Penha) no TJDFT."
)
CLARIFY_MESSAGE = (
  "Preciso de um esclarecimento: sua pergunta envolve família "
  "(guarda, visitas, alimentos, divórcio) ou violência doméstica?"
)
NO_BASIS_MESSAGE = "Não encontrei base segura no corpus para responder com citação."
CONTEXT_HITS = 2
DOMAIN_CLEAR_MIN = 0.6
TOPIC_HINTS = "Ex.: medidas protetivas, descumprimento, guarda, visitas, alimentos, divórcio."
ADVICE_RE = re.compile(
  r"\b(?:você deve|recomendo|aconselho|garanto que|causa ganha)\b",
  re.IGNORECASE,
)


@dataclass(frozen=True)
class Answer:
  """A deliverable response with its decision and verifiable sources."""

  decision: str
  text: str
  sources: tuple[str, ...]


def _relevant(query: str, hit: Hit) -> bool:
  """Content query token or citation must match the hit excerpt/reference."""
  query_tokens = set(content_tokens(query))
  excerpt_tokens = set(content_tokens(hit.snippet)) | set(content_tokens(hit.origin_ref))
  if query_tokens & excerpt_tokens:
    return True
  query_cites = set(extract_citations(query))
  hit_cites = set(extract_citations(hit.origin_ref)) | set(hit.citations)
  return bool(query_cites & hit_cites)


def verify_deliverable(text: str, sources: tuple[str, ...]) -> bool:
  """Automated pre-delivery checklist: sources, disclaimer, privacy, no advice."""
  return (
    bool(sources)
    and DISCLAIMER in text
    and not contains_personal_data(text)
    and not ADVICE_RE.search(text)
  )


def _deliver_or_defer(action: str, text: str, sources: tuple[str, ...]) -> Answer:
  if not verify_deliverable(text, sources):
    return Answer(decision="defer", text=NO_BASIS_MESSAGE, sources=())
  return Answer(decision=action, text=text, sources=sources)


def _render_accept(hit: Hit, context: list[Hit]) -> str:
  lines = [hit.origin_ref + ":", "", anonymize(hit.snippet)]
  if context:
    refs = "; ".join(item.origin_ref for item in context)
    lines += ["", f"Contexto: {refs}"]
  lines += ["", DISCLAIMER]
  return "\n".join(lines)


def _render_chain(chain: Chain) -> str:
  premise = chain.premises[0]
  facts = "; ".join(f'{fact.field}="{fact.span}"' for fact in chain.facts)
  lines = [
    f"Conclusão (cadeia verificada, {chain.rule_id}): {chain.conclusion}",
    f"Fatos: {facts}",
    "",
    f"{premise.ref}:",
    anonymize(premise.text),
    "",
    DISCLAIMER,
  ]
  return "\n".join(lines)


def _clarification(probabilities: Mapping[str, float]) -> str:
  """Domain-clear defers ask for the topic; ambiguous ones ask for the domain."""
  top = max(probabilities, key=lambda state: probabilities[state])
  if top != OUT_OF_SCOPE.code and probabilities[top] >= DOMAIN_CLEAR_MIN:
    return f"Sua pergunta parece ser sobre {AREAS[top].name}. Qual o tema específico? {TOPIC_HINTS}"
  return CLARIFY_MESSAGE


def _lacks_topic(query: str) -> bool:
  """True when the question names a law but no subject or article number."""
  if any(ref.startswith("art. ") for ref in extract_citations(query)):
    return False
  words = set(content_tokens(query)) - QUERY_FRAMING
  return any(words and words <= set(content_tokens(name)) for name in TOPICLESS_NAMES)


def compose(
  query: str,
  probabilities: Mapping[str, float],
  storage: CorpusStorage,
  encoder: Encoder | None = None,
) -> Answer:
  """Classify → decide → reason/retrieve → cited excerpt, or degrade honestly."""
  decision = decide(probabilities)
  if decision.action == "reject":
    return Answer(decision="reject", text=SCOPE_MESSAGE, sources=())
  if decision.action == "defer" or _lacks_topic(query):
    return Answer(decision="defer", text=_clarification(probabilities), sources=())
  area = decision.action.removeprefix("accept-")
  chain = build_chain(query, storage, area=area)
  if chain is not None:
    if not chain.complete:
      return Answer(decision="defer", text=NO_BASIS_MESSAGE, sources=())
    return _deliver_or_defer(
      decision.action,
      _render_chain(chain),
      (chain.conclusion_ref,),
    )
  # Statutes answer; decisions are used only when no statute matches.
  limit = CONTEXT_HITS + 1
  hits = search_statutes(storage, query, limit, encoder) or search(storage, query, limit)
  if not hits or not _relevant(query, hits[0]):
    return Answer(decision="defer", text=NO_BASIS_MESSAGE, sources=())
  top, context = hits[0], hits[1 : CONTEXT_HITS + 1]
  return _deliver_or_defer(
    decision.action,
    _render_accept(top, context),
    tuple(hit.origin_ref for hit in hits),
  )
