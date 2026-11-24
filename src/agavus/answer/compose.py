"""Answer composition: extractive, cited, honest (docs/response-quality.md).

Templates only — no generation, so no advice language and no ungrounded
claims by construction. Degradation ladder: cited answer → honest refusal
→ clarification. Sources are never empty on accept.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from agavus.corpus.anonymize import anonymize
from agavus.corpus.storage import CorpusStorage
from agavus.decide.rule import decide
from agavus.reason.chain import Chain, build_chain
from agavus.retrieve.search import WORD_RE, Hit, search
from agavus.taxonomy.lexicon import normalize

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


@dataclass(frozen=True)
class Answer:
  """A deliverable response with its decision and verifiable sources."""

  decision: str
  text: str
  sources: tuple[str, ...]


def _relevant(query: str, hit: Hit) -> bool:
  """At least one content query token must appear in the excerpt."""
  query_tokens = set(WORD_RE.findall(normalize(query)))
  excerpt_tokens = set(WORD_RE.findall(normalize(hit.snippet)))
  return bool(query_tokens & excerpt_tokens)


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


def compose(query: str, probabilities: Mapping[str, float], storage: CorpusStorage) -> Answer:
  """Classify → decide → reason/retrieve → cited excerpt, or degrade honestly."""
  decision = decide(probabilities)
  if decision.action == "reject":
    return Answer(decision="reject", text=SCOPE_MESSAGE, sources=())
  if decision.action == "defer":
    return Answer(decision="defer", text=CLARIFY_MESSAGE, sources=())
  chain = build_chain(query, storage)
  if chain is not None:
    if not chain.complete:
      return Answer(decision="defer", text=NO_BASIS_MESSAGE, sources=())
    return Answer(
      decision=decision.action,
      text=_render_chain(chain),
      sources=(chain.conclusion_ref,),
    )
  hits = search(storage, query, limit=CONTEXT_HITS + 1)
  if not hits or not _relevant(query, hits[0]):
    return Answer(decision="defer", text=NO_BASIS_MESSAGE, sources=())
  top, context = hits[0], hits[1 : CONTEXT_HITS + 1]
  return Answer(
    decision=decision.action,
    text=_render_accept(top, context),
    sources=tuple(hit.origin_ref for hit in hits),
  )
