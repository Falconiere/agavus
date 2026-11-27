"""Exercise the REST endpoints over a loopback socket with a real CPU model."""

import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Thread
from wsgiref.simple_server import make_server
from wsgiref.types import WSGIApplication

import httpx
import pytest
import torch

from agavus.model.agathe import END_ID, AgatheConfig, AgatheModel, framed, start
from agavus.model.law_search import search
from agavus.model.report import SAMPLE_CHARS
from agavus.model.sft import prompt, render_turns
from agavus.model.sft_context import NOT_FOUND, SEARCHED, with_context
from agavus.model.tests.law_search_test import LAW, NAME, synthetic_index
from agavus.model.tokenizer import encode, train_tokenizer
from agavus.server.api import LOOP_REPEATS, LOOPED, application, chat_prefix, looping

TOKENIZER = train_tokenizer(["um papel ção", "papel"] * 3, 300)
CONFIG = AgatheConfig(width=8, heads=2, kv_heads=1, context=4, layers=1)
MESSAGES = [
  {"role": "user", "text": "um papel"},
  {"role": "assistant", "text": "papel"},
  {"role": "user", "text": "um"},
]
TURNS = "<|user|>um papel<|end|><|assistant|>papel<|end|><|user|>um<|end|><|assistant|>"


@contextmanager
def serving(app: WSGIApplication) -> Iterator[httpx.Client]:
  """A loopback server for `app` and a client pointed at it."""
  with make_server("127.0.0.1", 0, app) as server:
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
      with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as client:
        yield client
    finally:
      server.shutdown()
      thread.join()


@pytest.fixture
def api() -> Iterator[httpx.Client]:
  with serving(application(start(TOKENIZER, CONFIG), "pequeno")) as client:
    yield client


def test_model_and_stream(api: httpx.Client) -> None:
  metadata = api.get("/api/model").json()
  assert metadata["name"] == "pequeno" and metadata["parameters"] > 0
  assert metadata["chat"] is False
  prefix = {"prefix": "um papel, outro papel e mais um papel"}
  with api.stream("POST", "/api/continue", json=prefix) as response:
    assert response.headers["content-type"] == "application/x-ndjson"
    events = [json.loads(line) for line in response.iter_lines()]
  metrics = events.pop()["metrics"]
  assert metrics["seconds"] > 0
  assert metrics["used"] == metrics["capacity"] == metadata["context"]
  assert metrics["characters"] == len("".join(event["text"] for event in events)) <= SAMPLE_CHARS


def test_invalid_requests(api: httpx.Client) -> None:
  assert api.get("/missing").is_client_error
  for body in [None, {}, {"prefix": 2}]:
    assert api.post("/api/continue", json=body).is_client_error


def test_missing_run_prevents_api_start(tmp_path: Path) -> None:
  result = subprocess.run(
    [sys.executable, "-m", "agavus.server", "--run", str(tmp_path)],
    capture_output=True,
    text=True,
    check=False,
  )
  assert result.returncode == 1 and str(tmp_path / "run.json") in result.stderr


def test_browser_page_and_assets_share_the_api_origin(api: httpx.Client) -> None:
  page = api.get("/")
  assert page.headers["content-type"].startswith("text/html")
  for asset, kind in (("chat.css", "text/css"), ("chat.js", "text/javascript")):
    assert f'"/{asset}"' in page.text
    response = api.get(f"/{asset}")
    assert response.status_code == httpx.codes.OK and response.content
    assert response.headers["content-type"].startswith(kind)
  assert api.get("/run.json").is_client_error


def _lines(app: WSGIApplication, messages: list[dict[str, str]]) -> list[str]:
  """Every JSON line `app` streams for one chat request."""
  body = {"messages": messages}
  with serving(app) as client, client.stream("POST", "/api/continue", json=body) as response:
    return list(response.iter_lines())


def _silent(context: int, favorite: int = END_ID) -> AgatheModel:
  """A model that always picks `favorite`; by default it answers nothing (the end token)."""
  model = start(TOKENIZER, replace(CONFIG, context=context))
  # With the residual paths silenced, only the token table decides.
  silenced = [weight for layer in model.layers for weight in (layer.mix, layer.down)]
  with torch.no_grad():
    for weight in silenced:
      weight.zero_()
    model.tokens.fill_(10.0)
    model.tokens[favorite].fill_(30.0)
  return model


def test_chat_run_sends_turns_in_turn_format_and_stops_at_the_end_token() -> None:
  model = _silent(64)
  assert chat_prefix(MESSAGES) == (TURNS, [])
  with serving(application(model, "conversa", chat=True)) as client:
    assert client.get("/api/model").json()["chat"] is True
    with client.stream("POST", "/api/continue", json={"messages": MESSAGES}) as response:
      events = [json.loads(line) for line in response.iter_lines()]
    for body in [{"prefix": "um"}, {"messages": MESSAGES[:2]}, {"messages": [{"role": "x"}]}]:
      assert client.post("/api/continue", json=body).is_client_error
  assert len(events) == 1 and events[0]["metrics"]["characters"] == 0
  assert events[0]["metrics"]["used"] == len(framed(TOKENIZER, TURNS)) - 1
  assert events[0]["sources"] == []


def test_chat_with_a_search_index_answers_from_the_articles_found(tmp_path: Path) -> None:
  db = synthetic_index(tmp_path)
  question = "Quem fica com a guarda dos filhos?"
  messages = [*MESSAGES[:2], {"role": "user", "text": question}]
  hits = search(db, question)
  expected = render_turns([("user", "um papel"), ("assistant", "papel")])
  expected += prompt(with_context(question, hits))
  assert chat_prefix(messages, db) == (expected, hits)
  model = _silent(4096)
  app = application(model, "conversa", chat=True, db=db)
  answer, event = [json.loads(line) for line in _lines(app, messages)]
  # An empty answer is grounded in nothing, so the chat defers.
  assert answer == {"text": NOT_FOUND}
  assert event["metrics"]["grounded"] is False and event["metrics"]["coverage"] == 0.0
  assert event["metrics"]["used"] == len(framed(TOKENIZER, expected)) - 1
  assert event["sources"][0] == {"name": NAME, "number": "2º", "origin_ref": f"{LAW}, art. 2"}
  assert len(event["sources"]) == len(hits) <= SEARCHED


def test_a_looping_answer_stops_with_a_notice() -> None:
  assert looping([7] * LOOP_REPEATS) and looping([1, 2, 3] + [4, 5] * LOOP_REPEATS)
  assert not looping([7] * (LOOP_REPEATS - 1)) and not looping(list(range(40)))
  favorite = encode(TOKENIZER, "papel")[0]
  app = application(_silent(64, favorite), "eco", chat=True)
  events = [json.loads(line) for line in _lines(app, MESSAGES)]
  texts = [event["text"] for event in events if "text" in event]
  assert texts[-1] == LOOPED and "papel" in "".join(texts[:-1])
  assert events[-1]["metrics"]["looped"] is True


def test_a_looping_answer_from_the_articles_defers_without_streaming(tmp_path: Path) -> None:
  db = synthetic_index(tmp_path)
  favorite = encode(TOKENIZER, "papel")[0]
  app = application(_silent(4096, favorite), "eco", chat=True, db=db)
  question = [{"role": "user", "text": "Quem fica com a guarda dos filhos?"}]
  answer, event = [json.loads(line) for line in _lines(app, question)]
  assert answer == {"text": NOT_FOUND}
  assert event["metrics"]["looped"] is True and event["metrics"]["grounded"] is False
  assert event["metrics"]["characters"] == len(NOT_FOUND)
