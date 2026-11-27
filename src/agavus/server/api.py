"""Loopback CPU inference API for the local chat. Conversations stay in the browser."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from secrets import randbits
from time import perf_counter
from wsgiref.types import StartResponse, WSGIApplication, WSGIEnvironment

from agavus.model.agathe import AgatheModel, framed
from agavus.model.report import SAMPLE_CHARS
from agavus.model.sft import ROLES, prompt, render_turns
from agavus.model.tokenizer import decode

MAX_BODY = 1024 * 1024
PARTIAL = "\ufffd"


def _line(value: dict[str, object]) -> bytes:
  return (json.dumps(value, ensure_ascii=False) + "\n").encode()


def continuation(model: AgatheModel, prefix: str) -> Iterator[bytes]:
  """Stream new text, then inference metrics including initial context usage in tokens."""
  capacity = model.config.context
  used = min(len(framed(model.tokenizer, prefix)) - 1, capacity)
  written: list[int] = []
  sent = ""
  seconds = 0.0
  tokens = model.continuation(prefix, seed=randbits(63), max_tokens=SAMPLE_CHARS)
  while True:
    began = perf_counter()
    token = next(tokens, None)
    seconds += perf_counter() - began
    if token is not None:
      written.append(token)
    text = decode(model.tokenizer, written)[:SAMPLE_CHARS]
    # A character split across tokens waits for its last byte.
    if text.endswith(PARTIAL) and token is not None:
      continue
    if len(text) > len(sent):
      yield _line({"text": text[len(sent) :]})
      sent = text
    if token is None or len(sent) == SAMPLE_CHARS:
      break
  yield _line(
    {"metrics": {"seconds": seconds, "characters": len(sent), "used": used, "capacity": capacity}}
  )


def chat_prefix(messages: object) -> str:
  """Earlier turns in turn format, then the last user message as a prompt.

  Raises ValueError unless `messages` is a non-empty list of {"role", "text"} objects
  ending with the user's.
  """
  if not isinstance(messages, list) or not messages:
    raise ValueError("messages must be a non-empty list")
  turns = [(message["role"], message["text"]) for message in messages]
  if any(role not in ROLES or not isinstance(text, str) for role, text in turns):
    raise ValueError("each message needs a known role and a text")
  *earlier, (role, text) = turns
  if role != "user":
    raise ValueError("the last message must be the user's")
  return render_turns(earlier) + prompt(text)


def application(model: AgatheModel, name: str, *, chat: bool = False) -> WSGIApplication:
  """Serve checkpoint metadata and newline-delimited continuations from the same model.

  A chat-tuned run (`chat`) takes {"messages": [...]} and answers in turn format; others
  take {"prefix": "..."} and continue it.
  """
  assets = Path(__file__).with_name("static")
  if not assets.is_dir():
    assets = Path(__file__).resolve().parents[3] / "chat"
  files = {
    "/": (assets / "index.html", "text/html"),
    "/chat.js": (assets / "chat.js", "text/javascript"),
    "/chat.css": (assets / "chat.css", "text/css"),
  }
  info = _line(
    {"name": name, "parameters": model.size(), "context": model.config.context, "chat": chat}
  )

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
      body = json.loads(env["wsgi.input"].read(length))
      prefix = chat_prefix(body["messages"]) if chat else body["prefix"]
      if not isinstance(prefix, str):
        raise ValueError
    except (ValueError, TypeError, KeyError):
      respond("400 Bad Request", headers)
      wanted = "as mensagens da conversa" if chat else "um prefixo de texto"
      return [_line({"error": f"Envie {wanted} em JSON."})]
    respond("200 OK", [("Content-Type", "application/x-ndjson"), ("Cache-Control", "no-store")])
    return continuation(model, prefix)

  return handle
