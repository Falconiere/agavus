"""Maritaca S5 as Agathe's teacher (docs/llm-lab.md §9.5).

The teacher sees one public-domain article at a time and writes question-and-answer pairs
from it alone. The citation is never the model's: any "Fonte" it writes is dropped and the
law's own name and article number are appended. Laws held out for evaluation never reach
the teacher, and neither do DJEN texts.
"""

import argparse
import json
import random
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from agavus.config.settings import load_settings
from agavus.model.sft import SFT_DIR, Conversation, save_conversations
from agavus.model.sft_rules import (
  LawCard,
  article_number,
  law_of,
  load_rule_sources,
  rule_conversations,
  split_by_law,
)

ENDPOINT = "https://chat.maritaca.ai/api/chat/completions"
MODEL = "S5"
SOURCE = "teacher:maritaca-S5"
LICENSE = "Maritaca S5 output (terms §2) written from public-domain law (Lei 9.610/1998, 8º, IV)"
# R$ per million tokens at the standard (non-batch) price, checked 2026-11-27.
PRICE_IN, PRICE_OUT = 0.50, 2.50
MIN_ARTICLE_CHARS = 200
ATTEMPTS = 3
SYSTEM = (
  "Você é professor de Direito brasileiro. Usando SOMENTE o texto do artigo fornecido, escreva "
  "{pairs} pares de pergunta e resposta em português do Brasil. As perguntas devem soar como as "
  "de um cidadão; as respostas devem ser fiéis ao texto, completas e sem inventar nada que não "
  "esteja no artigo. Não cite a fonte nem o nome da lei nas respostas. "
  'Responda apenas com JSON: {{"pares":[{{"pergunta":"...","resposta":"..."}}]}}'
)
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
SOURCE_LINE = re.compile(r"\s*\(?Fonte:.*$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Teacher:
  """The HTTP client, the key (never shown) and how many pairs to ask per article."""

  client: httpx.Client
  key: str = field(repr=False)
  pairs: int = 3
  effort: str = "low"
  pause: float = 1.0


@dataclass
class Usage:
  """Tokens and calls spent so far, to report the cost of a run."""

  calls: int = 0
  failed: int = 0
  prompt_tokens: int = 0
  completion_tokens: int = 0

  def reais(self) -> float:
    """Cost at the standard price, in R$."""
    return (self.prompt_tokens * PRICE_IN + self.completion_tokens * PRICE_OUT) / 1e6


def request_body(teacher: Teacher, card: LawCard, text: str) -> dict[str, object]:
  """The chat-completions request for one article."""
  return {
    "model": MODEL,
    "reasoning_effort": teacher.effort,
    "max_tokens": 4000,
    "messages": [
      {"role": "system", "content": SYSTEM.format(pairs=teacher.pairs)},
      {"role": "user", "content": f"{card.name}\n\n{text}"},
    ],
  }


def parse_pairs(content: str) -> list[tuple[str, str]]:
  """(question, answer) pairs from the teacher's JSON, model-written sources removed."""
  found = json.loads(FENCE.sub("", content.strip()))
  pairs = found.get("pares") if isinstance(found, dict) else None
  if not isinstance(pairs, list):
    raise ValueError("expected an object with a 'pares' list")
  kept: list[tuple[str, str]] = []
  for pair in pairs:
    if not isinstance(pair, dict):
      raise ValueError("each pair must be an object")
    question, answer = pair.get("pergunta"), pair.get("resposta")
    if not isinstance(question, str) or not isinstance(answer, str):
      raise ValueError("each pair needs a 'pergunta' and a 'resposta' string")
    answer = SOURCE_LINE.sub("", answer).strip()
    if question.strip() and answer:
      kept.append((question.strip(), answer))
  return kept


def _ask(teacher: Teacher, body: dict[str, object], usage: Usage) -> str:
  """The teacher's reply text, retrying transient failures with a growing pause."""
  for attempt in range(ATTEMPTS):
    try:
      response = teacher.client.post(
        ENDPOINT, json=body, headers={"Authorization": f"Bearer {teacher.key}"}
      )
      response.raise_for_status()
      data = response.json()
      usage.calls += 1
      usage.prompt_tokens += int(data["usage"]["prompt_tokens"])
      usage.completion_tokens += int(data["usage"]["completion_tokens"])
      return str(data["choices"][0]["message"]["content"])
    except httpx.HTTPStatusError as error:
      if error.response.status_code < httpx.codes.INTERNAL_SERVER_ERROR and (
        error.response.status_code != httpx.codes.TOO_MANY_REQUESTS
      ):
        raise
    except httpx.TransportError:
      pass
    time.sleep(teacher.pause * 2**attempt)
  raise httpx.TransportError(f"teacher failed {ATTEMPTS} times")


def teach(
  teacher: Teacher, card: LawCard, article: tuple[str, str], usage: Usage
) -> list[Conversation]:
  """Conversations for one article, each answer ending with the law's own citation."""
  origin_ref, text = article
  pairs = parse_pairs(_ask(teacher, request_body(teacher, card, text), usage))
  citation = f"Fonte: {card.name}, art. {article_number(origin_ref, text)}."
  return [
    Conversation(
      turns=(("user", question), ("assistant", f"{answer}\n\n{citation}")),
      source=SOURCE,
      origin_ref=origin_ref,
      license=LICENSE,
    )
    for question, answer in pairs
  ]


def choose_articles(
  cards: Sequence[LawCard],
  articles_by_url: dict[str, list[tuple[str, str]]],
  held_out: set[str],
  *,
  limit: int,
  seed: int,
) -> list[tuple[LawCard, tuple[str, str]]]:
  """A seeded sample of substantive articles from laws that are not held out."""
  pool = [
    (card, article)
    for card in cards
    if card.url not in held_out
    for article in articles_by_url.get(card.url, [])
    if len(article[1]) >= MIN_ARTICLE_CHARS
  ]
  return random.Random(seed).sample(pool, min(limit, len(pool)))


def run_teach(arguments: Sequence[str]) -> int:
  """`teach --limit N --out FILE`: teacher conversations from laws used for training only."""
  parser = argparse.ArgumentParser(prog="python -m agavus.model teach")
  parser.add_argument("--limit", type=int, default=50, help="articles to send (default: 50)")
  parser.add_argument("--pairs", type=int, default=3, help="pairs per article")
  parser.add_argument("--seed", type=int, default=0, help="which articles are sampled")
  parser.add_argument("--workers", type=int, default=4, help="requests in flight")
  parser.add_argument("--out", type=Path, help="JSONL (default: data/lab/sft/teacher-pilot.jsonl)")
  options = parser.parse_args(list(arguments))
  settings = load_settings()
  if not settings.maritaca_api_key:
    print("lab: set MARITACA_API_KEY (see .env.example)")
    return 1
  cards, articles = load_rule_sources(settings.data_dir)
  _, held_out = split_by_law(rule_conversations(cards, articles, seed=0))
  chosen = choose_articles(
    cards,
    articles,
    {law_of(conversation) for conversation in held_out},
    limit=options.limit,
    seed=options.seed,
  )
  usage = Usage()
  conversations: list[Conversation] = []
  with httpx.Client(timeout=180) as client, ThreadPoolExecutor(options.workers) as pool:
    teacher = Teacher(client, settings.maritaca_api_key, pairs=options.pairs)
    jobs = [pool.submit(teach, teacher, card, article, usage) for card, article in chosen]
    for job in as_completed(jobs):
      try:
        conversations.extend(job.result())
      except (httpx.HTTPError, ValueError, KeyError) as error:
        usage.failed += 1
        print(f"teach: skipped one article: {type(error).__name__}: {error}")
  out = options.out or settings.data_dir / SFT_DIR / "teacher-pilot.jsonl"
  save_conversations(out, sorted(conversations, key=lambda c: c.origin_ref))
  print(
    f"teach: {len(chosen)} articles, {usage.calls} calls, {usage.failed} failed, "
    f"{len(conversations)} conversations; tokens {usage.prompt_tokens:,} in / "
    f"{usage.completion_tokens:,} out ≈ R$ {usage.reais():.2f}\nsaved: {out}"
  )
  return 0
