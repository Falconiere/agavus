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

from agavus.model.agathe import END_ID, AgatheConfig, framed, start
from agavus.model.report import SAMPLE_CHARS
from agavus.model.tokenizer import train_tokenizer
from agavus.server.api import application, chat_prefix

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


def test_chat_run_sends_turns_in_turn_format_and_stops_at_the_end_token() -> None:
  model = start(TOKENIZER, replace(CONFIG, context=64))
  # With the residual paths silenced, every token scores the end token highest.
  silenced = [weight for layer in model.layers for weight in (layer.mix, layer.down)]
  with torch.no_grad():
    for weight in silenced:
      weight.zero_()
    model.tokens.fill_(10.0)
    model.tokens[END_ID].fill_(20.0)
  assert chat_prefix(MESSAGES) == TURNS
  with serving(application(model, "conversa", chat=True)) as client:
    assert client.get("/api/model").json()["chat"] is True
    with client.stream("POST", "/api/continue", json={"messages": MESSAGES}) as response:
      events = [json.loads(line) for line in response.iter_lines()]
    for body in [{"prefix": "um"}, {"messages": MESSAGES[:2]}, {"messages": [{"role": "x"}]}]:
      assert client.post("/api/continue", json=body).is_client_error
  assert len(events) == 1 and events[0]["metrics"]["characters"] == 0
  assert events[0]["metrics"]["used"] == len(framed(TOKENIZER, TURNS)) - 1
