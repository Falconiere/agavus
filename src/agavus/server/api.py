"""Loopback CPU inference API for the local chat. Conversations stay in the browser."""

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from wsgiref.types import StartResponse, WSGIApplication, WSGIEnvironment

from agavus.model.agathe import AgatheModel, framed
from agavus.model.generation import CHAT_CHARS, Tally, pieces
from agavus.model.grounding import checked, coverage
from agavus.model.law_search import Hit, search
from agavus.model.report import SAMPLE_CHARS
from agavus.model.sft import ROLES, prompt, render_turns
from agavus.model.sft_context import NOT_FOUND, SEARCHED, with_context

MAX_BODY = 1024 * 1024
LOOPED = "\n\n[Resposta interrompida: o modelo entrou em repetição. Confira os textos citados.]"


def _line(value: dict[str, object]) -> bytes:
  return (json.dumps(value, ensure_ascii=False) + "\n").encode()


def _closing(
  model: AgatheModel, prefix: str, sources: Sequence[Hit], shown: str, extra: dict[str, object]
) -> bytes:
  """The last line: inference metrics with initial context use in tokens, and the sources."""
  capacity = model.config.context
  used = min(len(framed(model.tokenizer, prefix)) - 1, capacity)
  metrics = {"characters": len(shown), "used": used, "capacity": capacity} | extra
  cited = [
    {"name": hit.name, "number": hit.number, "origin_ref": hit.origin_ref} for hit in sources
  ]
  return _line({"metrics": metrics, "sources": cited})


def continuation(
  model: AgatheModel,
  prefix: str,
  sources: Sequence[Hit] = (),
  max_chars: int = SAMPLE_CHARS,
  greedy: bool = False,
) -> Iterator[bytes]:
  """Stream new text, then inference metrics including initial context usage in tokens.

  A loop of repeated tokens ends the text with the LOOPED notice. The last line also lists
  the articles placed in context (`sources`). `greedy` decodes the likeliest tokens.
  """
  tally = Tally()
  sent = ""
  for piece in pieces(model, prefix, max_chars, greedy, tally):
    sent += piece
    yield _line({"text": piece})
  if tally.looped:
    yield _line({"text": LOOPED})
  yield _closing(model, prefix, sources, sent, {"seconds": tally.seconds, "looped": tally.looped})


def grounded_answer(
  model: AgatheModel, prefix: str, sources: Sequence[Hit], question: str
) -> Iterator[bytes]:
  """The whole greedy answer once it passes the grounding check, else the NOT_FOUND deferral.

  Nothing streams before the check, so an invented answer is never shown. Metrics add
  `grounded` and the answer's `coverage` of the shown articles; a looped answer defers.
  """
  tally = Tally()
  answer = "".join(pieces(model, prefix, CHAT_CHARS, True, tally))
  shown = NOT_FOUND if tally.looped else checked(answer, sources, question)
  extra: dict[str, object] = {"seconds": tally.seconds, "looped": tally.looped}
  extra |= {"grounded": shown == answer, "coverage": coverage(answer, sources)}
  yield _line({"text": shown})
  yield _closing(model, prefix, sources, shown, extra)


def chat_prefix(messages: object, db: Path | None = None) -> tuple[str, list[Hit]]:
  """Earlier turns in turn format, then the last user message as a prompt.

  With a search index `db`, the last message is sent with the articles found for it, which
  are returned too. Raises ValueError unless `messages` is a non-empty list of
  {"role", "text"} objects ending with the user's.
  """
  if not isinstance(messages, list) or not messages:
    raise ValueError("messages must be a non-empty list")
  turns = [(message["role"], message["text"]) for message in messages]
  if any(role not in ROLES or not isinstance(text, str) for role, text in turns):
    raise ValueError("each message needs a known role and a text")
  *earlier, (role, text) = turns
  if role != "user":
    raise ValueError("the last message must be the user's")
  hits = [] if db is None else search(db, text, SEARCHED)
  question = text if db is None else with_context(text, hits)
  return render_turns(earlier) + prompt(question), hits


def application(
  model: AgatheModel, name: str, *, chat: bool = False, db: Path | None = None
) -> WSGIApplication:
  """Serve checkpoint metadata and newline-delimited continuations from the same model.

  A chat-tuned run (`chat`) takes {"messages": [...]} and answers greedily in turn format;
  with a search index `db`, from the articles found, shown only when grounded in them.
  Others take {"prefix": "..."} and sample.
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
      prefix, sources = chat_prefix(body["messages"], db) if chat else (body["prefix"], [])
      if not isinstance(prefix, str):
        raise ValueError
    except (ValueError, TypeError, KeyError):
      respond("400 Bad Request", headers)
      wanted = "as mensagens da conversa" if chat else "um prefixo de texto"
      return [_line({"error": f"Envie {wanted} em JSON."})]
    respond("200 OK", [("Content-Type", "application/x-ndjson"), ("Cache-Control", "no-store")])
    if chat and db is not None:
      return grounded_answer(model, prefix, sources, body["messages"][-1]["text"])
    return continuation(model, prefix, sources, CHAT_CHARS if chat else SAMPLE_CHARS, chat)

  return handle
