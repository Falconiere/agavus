"""Verified syllogism chains: facts + rule + verbatim premise.

A chain is complete only when the conclusion's provision exists verbatim in
the corpus. Broken chains never ship (docs/reasoning.md guarantees).
"""

from dataclasses import dataclass

from agavus.corpus.storage import CorpusStorage
from agavus.reason.facts import Fact, extract_facts
from agavus.reason.rules import RULES, Rule


@dataclass(frozen=True)
class Premise:
  """A verbatim provision supporting the conclusion."""

  ref: str
  text: str


@dataclass(frozen=True)
class Chain:
  """One verified (or broken) reasoning chain, fully auditable."""

  rule_id: str
  facts: tuple[Fact, ...]
  conclusion: str
  conclusion_ref: str
  premises: tuple[Premise, ...]
  complete: bool


def _matching_rule(fields: set[str]) -> Rule | None:
  for rule in RULES:
    if all(condition in fields for condition in rule.conditions):
      return rule
  return None


def _premise(storage: CorpusStorage, ref: str) -> Premise | None:
  for document in storage.load_texts():
    if document.origin_ref == ref:
      return Premise(ref=ref, text=document.text)
  return None


def build_chain(query: str, storage: CorpusStorage) -> Chain | None:
  """Fire the first fully-grounded rule; None when no rule applies."""
  facts = extract_facts(query)
  rule = _matching_rule({fact.field for fact in facts})
  if rule is None:
    return None
  used = tuple(fact for fact in facts if fact.field in rule.conditions)
  premise = _premise(storage, rule.conclusion_ref)
  premises = (premise,) if premise is not None else ()
  return Chain(
    rule_id=rule.id,
    facts=used,
    conclusion=rule.conclusion,
    conclusion_ref=rule.conclusion_ref,
    premises=premises,
    complete=premise is not None,
  )
