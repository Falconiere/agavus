"""Start the local Agathe REST service with python -m agavus.server."""

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server

import torch

from agavus.config.settings import load_settings
from agavus.model.bundle import BUNDLE_FILE
from agavus.model.checkpoint import load_checkpoint
from agavus.model.law_search import SEARCH_FILE
from agavus.model.report import THREADS
from agavus.model.tokenizer import TOKENIZER_FILE
from agavus.server.api import application


class LocalServer(ThreadingMixIn, WSGIServer):
  """Serve browser assets while other connections are idle or generating replies."""

  daemon_threads = True


def serve(arguments: Sequence[str]) -> int:
  """Serve the static chat and CPU inference API on one loopback origin."""
  data_dir = load_settings().data_dir
  parser = argparse.ArgumentParser(prog="python -m agavus.server")
  parser.add_argument("--port", type=int, default=7861, help="porta local (padrão: 7861)")
  parser.add_argument(
    "--run", type=Path, default=data_dir / "lab/runs/long-civil", help="diretório do checkpoint"
  )
  parser.add_argument(
    "--tokenizer", type=Path, default=data_dir / TOKENIZER_FILE, help="tokenizer do checkpoint"
  )
  parser.add_argument(
    "--search", type=Path, default=data_dir / SEARCH_FILE, help="índice de busca dos artigos"
  )
  options = parser.parse_args(arguments)
  run: Path = options.run
  torch.set_num_threads(THREADS)
  try:
    model = load_checkpoint(run, data_dir / BUNDLE_FILE, options.tokenizer, data_dir)
    # Chat-tuned runs (`python -m agavus.model sft`) answer in turn format.
    chat = json.loads((run / "run.json").read_bytes()).get("chat") is True
  except (OSError, ValueError) as error:
    parser.exit(1, f"{error}\n")
  # Without an index, a chat run answers from memory as before.
  db = options.search if chat and options.search.is_file() else None
  app = application(model, run.resolve().name, chat=chat, db=db)
  with make_server("127.0.0.1", options.port, app, server_class=LocalServer) as server:
    print(f"Agathe · http://127.0.0.1:{server.server_port}", flush=True)
    with suppress(KeyboardInterrupt):
      server.serve_forever()
  return 0


if __name__ == "__main__":
  sys.exit(serve(sys.argv[1:]))
