"""Maritaca teacher on synthetic replies: request shape, parsing, our citation, retries."""

import json
from collections.abc import Callable

import httpx
import pytest

from agavus.corpus.httpfixtures import mock_client
from agavus.model.sft_rules import LawCard
from agavus.model.teacher import (
  ENDPOINT,
  LICENSE,
  MIN_ARTICLE_CHARS,
  MODEL,
  SOURCE,
  Teacher,
  Usage,
  choose_articles,
  parse_pairs,
  request_body,
  teach,
)

CARD = LawCard("https://planalto/l1.htm", "Lei nº 1, de 2 de janeiro de 2030", "Cria regras.")
ARTICLE = ("https://planalto/l1.htm, art. 5", "Art. 5º O sindicato precisa de um terço dos sócios.")
PAIRS = {
  "pares": [
    {"pergunta": "Quantos sócios?", "resposta": "Um terço. Fonte: CLT, art. 5."},
    {"pergunta": "Quem decide?", "resposta": "O sindicato."},
  ]
}


def _reply(content: str) -> httpx.Response:
  usage = {"prompt_tokens": 100, "completion_tokens": 400}
  return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": usage})


def _server(
  replies: list[httpx.Response],
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
  """A handler answering with `replies` in order (the last one repeats), and its requests."""
  calls: list[httpx.Request] = []

  def serve(request: httpx.Request) -> httpx.Response:
    calls.append(request)
    return replies[min(len(calls), len(replies)) - 1]

  return calls, serve


def test_request_sends_the_law_name_and_article_to_s5_at_low_effort() -> None:
  with mock_client(lambda _: httpx.Response(500)) as client:
    body = request_body(Teacher(client, "secret", pairs=4), CARD, ARTICLE[1])
  messages = body["messages"]
  assert (body["model"], body["reasoning_effort"]) == (MODEL, "low")
  assert isinstance(messages, list)
  system = messages[0]["content"]
  assert "4 pares" in system
  assert "compreensíveis sozinhas" in system
  assert "nome completo" in system
  assert "assinatura" in system
  assert messages[1]["content"] == f"{CARD.name}\n\n{ARTICLE[1]}"


def test_parse_accepts_fenced_json_and_drops_model_sources() -> None:
  fenced = "```json\n" + json.dumps(PAIRS, ensure_ascii=False) + "\n```"
  assert parse_pairs(fenced) == [("Quantos sócios?", "Um terço."), ("Quem decide?", "O sindicato.")]
  for bad in ('{"pares": "x"}', '{"pares": [{"pergunta": 1}]}', "[]", "não é JSON"):
    with pytest.raises(ValueError):
      parse_pairs(bad)


def test_teach_appends_our_citation_and_provenance_after_a_retry() -> None:
  calls, serve = _server([httpx.Response(503), _reply(json.dumps(PAIRS, ensure_ascii=False))])
  usage = Usage()
  with mock_client(serve) as client:
    found = teach(Teacher(client, "secret", pause=0), CARD, ARTICLE, usage)
  assert [str(request.url) for request in calls] == [ENDPOINT, ENDPOINT]
  assert calls[0].headers["authorization"] == "Bearer secret"
  assert found[0].turns == (
    ("user", "Quantos sócios?"),
    ("assistant", f"Um terço.\n\nFonte: {CARD.name}, art. 5º."),
  )
  assert {(c.source, c.origin_ref, c.license) for c in found} == {(SOURCE, ARTICLE[0], LICENSE)}
  assert (usage.calls, usage.prompt_tokens, usage.completion_tokens) == (1, 100, 400)
  assert usage.reais() > 0


def test_client_errors_are_not_retried_and_the_key_is_never_shown() -> None:
  calls, serve = _server([httpx.Response(401)])
  with mock_client(serve) as client:
    teacher = Teacher(client, "secret", pause=0)
    with pytest.raises(httpx.HTTPStatusError):
      teach(teacher, CARD, ARTICLE, Usage())
  assert len(calls) == 1
  assert "secret" not in repr(teacher)


def test_articles_come_only_from_training_laws_and_are_substantive() -> None:
  other = LawCard("https://planalto/l2.htm", "Lei nº 2, de 3 de março de 2030", "Outra.")
  long_text = "Art. 1º " + "regra " * MIN_ARTICLE_CHARS
  articles = {
    CARD.url: [("l1, art. 1", long_text), ("l1, art. 2", "Art. 2º Curto.")],
    other.url: [("l2, art. 1", long_text)],
  }
  chosen = choose_articles([CARD, other], articles, {other.url}, limit=10, seed=1)
  assert chosen == [(CARD, ("l1, art. 1", long_text))]
  assert choose_articles([CARD, other], articles, set(), limit=1, seed=1) == choose_articles(
    [CARD, other], articles, set(), limit=1, seed=1
  )
