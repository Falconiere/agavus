"""Pre-upload privacy scans (docs/publishing.md: zero tolerance).

Uploads are public; every staging tree is scanned for personal data before
it reaches Hugging Face. HTML is reduced to prose first (tags and URL paths
are not prose), and path-like tokens are ignored elsewhere so statute URLs
never trip the phone pattern.
"""

import re
from collections.abc import Iterable
from pathlib import Path

from agavus.corpus.anonymize import contains_personal_data
from agavus.corpus.statutes import decode_html, html_to_text

TEXT_SUFFIXES = frozenset(
  {".htm", ".html", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
)
HTML_SUFFIXES = frozenset({".htm", ".html"})
SKIP_PARTS = frozenset({"__pycache__", ".pytest_cache", "tests"})
_PATH_TOKEN = re.compile(r"(?:https?://|\.{0,2}/)[^\s\"'<>]*")


def readable_text(path: Path) -> str:
  """Text the scan inspects: HTML as prose, other files with paths masked."""
  payload = path.read_bytes()
  if path.suffix in HTML_SUFFIXES:
    return html_to_text(decode_html(payload))
  return _PATH_TOKEN.sub(" ", payload.decode("utf-8", errors="ignore"))


def personal_data_paths(paths: Iterable[Path]) -> list[Path]:
  """Candidate files whose text trips the high-confidence privacy scan."""
  flagged: list[Path] = []
  for path in paths:
    if path.suffix not in TEXT_SUFFIXES or SKIP_PARTS.intersection(path.parts):
      continue
    if path.is_file() and contains_personal_data(readable_text(path)):
      flagged.append(path)
  return flagged


def scan_tree(root: Path) -> list[Path]:
  """Recursively scan a staging directory; empty result means clean."""
  return personal_data_paths(sorted(root.rglob("*")))


def ensure_no_personal_data(root: Path) -> None:
  """Raise ValueError when any scanned file under root carries personal data."""
  problems = scan_tree(root)
  if problems:
    names = ", ".join(str(path) for path in problems)
    raise ValueError(f"personal data found (publication blocked): {names}")
