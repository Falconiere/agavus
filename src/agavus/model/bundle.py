"""Lab text in one file, so a rented machine never needs corpus.db or data/raw/.

docs/llm-lab.md §6: export the allowlisted text locally, inspect it, then
upload only that file. The SHA-256 identifies the data in every run record.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from agavus.model.data import load_articles
from agavus.model.extra import civil_code_rest, load_wikipedia
from agavus.model.federal import load_federal

BUNDLE_FILE = Path("lab") / "export" / "texts.json"


@dataclass(frozen=True)
class Texts:
  """Statute articles (split later, by article) and the named extra texts."""

  statutes: tuple[str, ...]
  extras: dict[str, tuple[str, ...]] = field(default_factory=dict)


def local_texts(data_dir: Path) -> Texts:
  """Everything the lab may train on, read from this machine's data dir."""
  return Texts(
    statutes=load_articles(data_dir),
    extras={"civil": civil_code_rest(data_dir), "wikipedia": load_wikipedia(data_dir)},
  )


def federal_texts(data_dir: Path) -> Texts:
  """Combine the frozen statute bundle with the local federal articles, without SQLite."""
  base = read_bundle(data_dir / BUNDLE_FILE)
  return Texts(base.statutes, {**base.extras, "federal": load_federal(data_dir)})


def digest(path: Path) -> str:
  """SHA-256 of a file, to compare copies and to name the data in run records."""
  return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle_bytes(texts: Texts) -> bytes:
  """Stable encoding used to identify the training inputs."""
  payload = {
    "statutes": list(texts.statutes),
    "extras": {k: list(v) for k, v in texts.extras.items()},
  }
  return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()


def write_bundle(path: Path, texts: Texts) -> str:
  """Write the texts as JSON and return their SHA-256."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(bundle_bytes(texts))
  return digest(path)


def read_bundle(path: Path) -> Texts:
  """Texts written by write_bundle."""
  payload = json.loads(path.read_text(encoding="utf-8"))
  extras = {str(name): tuple(map(str, values)) for name, values in payload["extras"].items()}
  return Texts(statutes=tuple(map(str, payload["statutes"])), extras=extras)
