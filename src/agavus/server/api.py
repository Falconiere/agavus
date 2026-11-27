"""Loopback CPU inference API for the local chat. Conversations stay in the browser."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from secrets import randbits
from time import perf_counter
from wsgiref.types import StartResponse, WSGIApplication, WSGIEnvironment

from agavus.model.agathe import AgatheModel
from agavus.model.report import SAMPLE_CHARS

MAX_BODY = 1024 * 1024


def _line(value: dict[str, object]) -> bytes:
  return (json.dumps(value, ensure_ascii=False) + "\n").encode()


def continuation(model: AgatheModel, prefix: str) -> Iterator[bytes]:
  """Stream new characters, then inference metrics including initial context usage."""
  capacity = model.config.context
  recent = prefix[-capacity:]
  count = 0
  seconds = 0.0
  tokens = model.continuation(recent, seed=randbits(63), max_chars=SAMPLE_CHARS)
  while True:
    began = perf_counter()
    token = next(tokens, None)
    seconds += perf_counter() - began
    if token is None:
      break
    text = model.vocab.decode([token])
    count += len(text)
    yield _line({"text": text})
  yield _line(
    {
      "metrics": {
        "seconds": seconds,
        "characters": count,
        "used": len(recent),
        "capacity": capacity,
      }
    }
  )


def application(model: AgatheModel, name: str) -> WSGIApplication:
  """Serve checkpoint metadata and newline-delimited continuations from the same model."""
  assets = Path(__file__).with_name("static")
  if not assets.is_dir():
    assets = Path(__file__).resolve().parents[3] / "chat"
  files = {
    "/": (assets / "index.html", "text/html"),
    "/chat.js": (assets / "chat.js", "text/javascript"),
    "/chat.css": (assets / "chat.css", "text/css"),
  }
  info = _line({"name": name, "parameters": model.size(), "context": model.config.context})

  def handle(env: WSGIEnvironment, respond: StartResponse) -> Iterable[bytes]:
    headers = [("Content-Type", "application/json"), ("Cache-Control", "no-store")]
    route = (env["REQUEST_METHOD"], env["PATH_INFO"])
    method, path = route
    if method == "GET" and path in files:
      file, kind = files[path]
      respond("200 OK", [("Content-Type", f"{kind}; charset=utf-8")])
      return [file.read_bytes()]
    if route == ("GET", "/api/model"):
      respond("200 OK", headers)
      return [info]
    if route != ("POST", "/api/continue"):
      respond("404 Not Found", headers)
      return [_line({"error": "Rota desconhecida."})]
    try:
      length = int(env.get("CONTENT_LENGTH") or 0)
      media_type = env.get("CONTENT_TYPE", "").partition(";")[0].strip().lower()
      if not 0 < length <= MAX_BODY or media_type != "application/json":
        raise ValueError
      prefix = json.loads(env["wsgi.input"].read(length))["prefix"]
      if not isinstance(prefix, str):
        raise ValueError
    except (ValueError, TypeError, KeyError):
      respond("400 Bad Request", headers)
      return [_line({"error": "Envie um prefixo de texto em JSON."})]
    respond("200 OK", [("Content-Type", "application/x-ndjson"), ("Cache-Control", "no-store")])
    return continuation(model, prefix)

  return handle
