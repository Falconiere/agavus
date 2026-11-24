"""Tests for the shared HTTP fixtures (tests folder)."""

import httpx

from agavus.corpus.httpfixtures import json_handler, mock_client, text_handler


def test_json_handler() -> None:
  http_ok = 200
  response = json_handler({"ok": True})(httpx.Request("GET", "https://example.test"))
  assert response.status_code == http_ok
  assert response.json() == {"ok": True}


def test_text_handler_and_mock_client() -> None:
  client = mock_client(text_handler("olá"))
  assert client.get("https://example.test").text == "olá"
