"""More training text for the lab (docs/llm-lab.md §3): licenses recorded there.

Two sources, both training-only:

- the rest of the Civil Code: the corpus keeps only the family book; the page
  already fetched holds every article. Public domain (Lei 9.610/1998, art. 8º, IV).
- Portuguese Wikipedia (CC BY-SA 4.0), first sections of articles, with each
  article's URL kept for attribution.

The exam stays the same statute articles. Any extra text that shares a long
run of characters with a statute article is dropped, so nothing leaks into it.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from agavus.corpus.statutes import STATUTES, segment_articles

CIVIL_CODE = "Lei 10.406/2002"
STATUTE_RAW = Path("raw") / "statutes"
WIKIPEDIA_FILE = Path("lab") / "text" / "wikipedia-pt.jsonl"
WIKIPEDIA_ROWS = "https://datasets-server.huggingface.co/rows"
WIKIPEDIA_DATASET = {"dataset": "wikimedia/wikipedia", "config": "20231101.pt", "split": "train"}
PAGE = 100
# Long enough that shared legal boilerplate ("Redação dada pela Lei ...") passes.
LEAK_RUN = 100
SHINGLE = 32
LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
MARKDOWN_HEADING = re.compile(r"#+")
OWN_ARTICLE = re.compile(r"\s(?=Art\.\s*\d)")


@dataclass(frozen=True)
class WikipediaArticle:
  """Opening of one article, with its URL for attribution."""

  title: str
  url: str
  text: str


def _article_number(number: str) -> int:
  return int(number.replace(".", "").split("-")[0])


def civil_code_rest(data_dir: Path) -> tuple[str, ...]:
  """Civil Code articles the corpus doesn't hold, from the newest saved page."""
  pages = sorted((data_dir / STATUTE_RAW).glob(f"*/{CIVIL_CODE.replace('/', '-')}.*"))
  if not pages:
    return ()
  kept = next(statute.article_range for statute in STATUTES if statute.name == CIVIL_CODE)
  low, high = kept or (0, -1)
  page = LINK.sub(r"\1", pages[-1].read_text(encoding="utf-8"))
  text = OWN_ARTICLE.sub("\n", MARKDOWN_HEADING.sub(" ", page))
  return tuple(
    " ".join(body.split())
    for number, body in segment_articles(text)
    if not low <= _article_number(number) <= high
  )


def fetch_wikipedia(
  client: httpx.Client, *, articles: int, chars: int, offset: int = 0
) -> tuple[WikipediaArticle, ...]:
  """The first `chars` characters of `articles` Wikipedia articles, in dataset order."""
  found: list[WikipediaArticle] = []
  while len(found) < articles:
    params = {**WIKIPEDIA_DATASET, "offset": str(offset + len(found)), "length": str(PAGE)}
    response = client.get(WIKIPEDIA_ROWS, params=params)
    response.raise_for_status()
    rows = response.json()["rows"]
    if not rows:
      break
    for entry in rows:
      row = entry["row"]
      found.append(WikipediaArticle(str(row["title"]), str(row["url"]), str(row["text"])[:chars]))
  return tuple(found[:articles])


def save_wikipedia(data_dir: Path, articles: Sequence[WikipediaArticle]) -> Path:
  """One JSON line per article under data/lab/ (gitignored)."""
  path = data_dir / WIKIPEDIA_FILE
  path.parent.mkdir(parents=True, exist_ok=True)
  lines = (
    json.dumps({"title": a.title, "url": a.url, "text": a.text}, ensure_ascii=False)
    for a in articles
  )
  path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
  return path


def load_wikipedia(data_dir: Path) -> tuple[str, ...]:
  """Saved Wikipedia texts; empty until fetched."""
  path = data_dir / WIKIPEDIA_FILE
  if not path.exists():
    return ()
  lines = path.read_text(encoding="utf-8").splitlines()
  return tuple(str(json.loads(line)["text"]) for line in lines if line)


def _runs(text: str) -> set[str]:
  flat = " ".join(text.split())
  return {flat[start : start + LEAK_RUN] for start in range(len(flat) - LEAK_RUN + 1)}


def without_leaks(texts: Sequence[str], protected: Sequence[str]) -> tuple[str, ...]:
  """Drop texts that share LEAK_RUN characters in a row with a protected text.

  A shared run covers a SHINGLE-long piece starting at some multiple of the stride, so only
  texts with such a piece in common get the exact (slow) comparison.
  """
  guarded: set[str] = set()
  for text in protected:
    guarded |= _runs(text)
  pieces = {
    flat[start : start + SHINGLE]
    for flat in (" ".join(text.split()) for text in protected)
    for start in range(len(flat) - SHINGLE + 1)
  }
  stride = LEAK_RUN - SHINGLE

  def suspect(text: str) -> bool:
    flat = " ".join(text.split())
    ends = range(0, len(flat) - LEAK_RUN + stride + 1, stride)
    return any(flat[start : start + SHINGLE] in pieces for start in ends)

  return tuple(text for text in texts if not suspect(text) or guarded.isdisjoint(_runs(text)))
