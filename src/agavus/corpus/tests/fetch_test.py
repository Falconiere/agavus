"""Tests for the polite fetcher (tests folder, mock transport, no network)."""

from datetime import date
from pathlib import Path

import httpx
import pytest

from agavus.corpus.fetch import Fetcher, FetchError, FetchPolicy, JsonRequest
from agavus.corpus.httpfixtures import Handler, json_handler, mock_client, text_handler
from agavus.corpus.storage import CorpusStorage


def _fetcher(tmp_path: Path, handler: Handler) -> Fetcher:
  policy = FetchPolicy(min_interval=0, backoff_base=0)
  return Fetcher(CorpusStorage(tmp_path), "datajud", client=mock_client(handler), policy=policy)


def test_success_preserves_raw(tmp_path: Path) -> None:
  fetcher = _fetcher(tmp_path, json_handler({"ok": True}))
  request = JsonRequest("POST", "https://example.test/_search", payload={"size": 1})
  body = fetcher.fetch(request)
  assert b'"ok"' in body
  assert fetcher.last_raw_ref.startswith("raw/datajud/")
  assert (tmp_path / fetcher.last_raw_ref).exists()


def test_retries_on_429_then_succeeds(tmp_path: Path) -> None:
  attempts_until_success = 3
  calls: list[int] = []

  def handler(request: httpx.Request) -> httpx.Response:
    calls.append(1)
    if len(calls) < attempts_until_success:
      return httpx.Response(429)
    return httpx.Response(200, text="ok")

  fetcher = _fetcher(tmp_path, handler)
  assert fetcher.fetch(JsonRequest("GET", "https://example.test/x")) == b"ok"
  assert len(calls) == attempts_until_success


def test_retry_wait_honors_retry_after(tmp_path: Path) -> None:
  expected_wait = 7.0
  fetcher = _fetcher(tmp_path, json_handler({}))
  response = httpx.Response(429, headers={"Retry-After": "7"})
  assert fetcher._retry_wait(0, response) == expected_wait


def test_retry_wait_capped_without_header(tmp_path: Path) -> None:
  policy = FetchPolicy(min_interval=0)
  fetcher = Fetcher(CorpusStorage(tmp_path), "datajud", policy=policy)
  wait = fetcher._retry_wait(10, None)
  cap = FetchPolicy().max_backoff
  assert cap / 2 <= wait <= cap


def test_zero_backoff_does_not_sleep(tmp_path: Path) -> None:
  policy = FetchPolicy(min_interval=0, backoff_base=0)
  fetcher = Fetcher(CorpusStorage(tmp_path), "datajud", policy=policy)
  assert fetcher._retry_wait(0, None) == 0.0


def test_gives_up_after_retries(tmp_path: Path) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(429)

  fetcher = _fetcher(tmp_path, handler)
  with pytest.raises(FetchError):
    fetcher.fetch(JsonRequest("GET", "https://example.test/x"))


def test_hard_error_not_retried(tmp_path: Path) -> None:
  calls: list[int] = []

  def handler(request: httpx.Request) -> httpx.Response:
    calls.append(1)
    return httpx.Response(401)

  fetcher = _fetcher(tmp_path, handler)
  with pytest.raises(FetchError):
    fetcher.fetch(JsonRequest("GET", "https://example.test/x"))
  assert len(calls) == 1


def test_fetch_json(tmp_path: Path) -> None:
  fetcher = _fetcher(tmp_path, text_handler('{"hits": {"hits": []}}'))
  data = fetcher.fetch_json(JsonRequest("POST", "https://example.test/_search"))
  assert "hits" in data
  assert date.today().isoformat() in fetcher.last_raw_ref
