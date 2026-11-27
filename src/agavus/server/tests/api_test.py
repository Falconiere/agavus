"""Exercise the REST endpoints over a loopback socket with a real CPU model."""

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from threading import Thread
from wsgiref.simple_server import make_server

import httpx
import pytest

from agavus.model.agathe import AgatheConfig, start
from agavus.model.bigram import build_vocab
from agavus.model.report import SAMPLE_CHARS
from agavus.server.api import application


@pytest.fixture
def api() -> Iterator[httpx.Client]:
  model = start(build_vocab(["papel"]), AgatheConfig(width=8, heads=2, context=4, layers=1))
  with make_server("127.0.0.1", 0, application(model, "pequeno")) as server:
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
      with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as client:
        yield client
    finally:
      server.shutdown()
      thread.join()


def test_model_and_stream(api: httpx.Client) -> None:
  metadata = api.get("/api/model").json()
  assert metadata["name"] == "pequeno" and metadata["parameters"] > 0
  with api.stream("POST", "/api/continue", json={"prefix": "um papel"}) as response:
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
