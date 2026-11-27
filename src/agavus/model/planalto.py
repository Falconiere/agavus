"""Every federal law Planalto indexes, in its newest consolidated text (docs/llm-lab.md §3).

Index pages lead to law pages. A law page that links a `compilado` or `cons` version is
replaced by that version; otherwise Planalto keeps the page itself up to date. Each capture
keeps the ETag and Last-Modified Planalto sent, so a rerun only downloads what changed.

The new legislation portal lists decree-laws but refuses non-browser clients: save its
listing pages under `indexes/` with a browser, and their links join the crawl.
"""

import json
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from agavus.corpus.statutes import decode_html
from agavus.model.extra import without_leaks
from agavus.model.federal import PUBLIC_DOMAIN, FederalLaw, articles, save_bytes

SITE = "https://www.planalto.gov.br/ccivil_03/"
AGENT = "Mozilla/5.0 (compatible; agavus-lab/0.1)"
PLANALTO_DIR = Path("lab/text/planalto")
INDEXES = (
  "leis/_Lei-Ordinaria.htm",
  "leis/lcp/quadro_lcp.htm",
  "leis/ldl/quadro_ldl.htm",
  "codigos/quadro_cod.htm",
)
FIXED = ("constituicao/constituicaocompilado.htm",)
INDEX = re.compile(r"(?:quadro[^/]*|/quadro/[^/]+|/_leis\d{4})\.htm$")
LAW = re.compile(r"/(?:l|lcp|ldl|del)[\d.]+[a-z]?\.htm$")
CONSOLIDATED = re.compile(r"(?:compilad[oa]|cons)\.htm$")
HREF = re.compile(r"""href\s*=\s*["']([^"'#]+)""", re.IGNORECASE)


@dataclass(frozen=True)
class Crawl:
  """How politely to fetch: pause between requests, and the client to use."""

  client: httpx.Client
  interval: float = 1.0


def links(html: str, base: str) -> Iterator[str]:
  """Absolute, lowercased ccivil_03 URLs on a page, mapping the new portal to Planalto."""
  for href in HREF.findall(html):
    parts = urlsplit(urljoin(base, href.strip()))
    path = parts.path.lower()
    marker = path.find("/ccivil_03/")
    if parts.scheme in {"http", "https"} and marker >= 0:
      yield SITE + path[marker + len("/ccivil_03/") :]


def _get(crawl: Crawl, url: str) -> str:
  time.sleep(crawl.interval)
  response = crawl.client.get(url, headers={"User-Agent": AGENT}, follow_redirects=True)
  response.raise_for_status()
  return decode_html(response.content)


def discover(crawl: Crawl, captured: Sequence[Path] = ()) -> list[str]:
  """Law page URLs reachable from the Planalto indexes and the saved portal listings."""
  queue = [SITE + path for path in INDEXES]
  seen = set(queue)
  laws = {SITE + path for path in FIXED}
  pages = [(path.read_text(encoding="utf-8"), SITE) for path in captured]
  while queue or pages:
    if pages:
      html, base = pages.pop()
    else:
      base = queue.pop(0)
      html = _get(crawl, base)
    for url in links(html, base):
      if LAW.search(url):
        laws.add(url)
      elif INDEX.search(url) and url not in seen:
        seen.add(url)
        queue.append(url)
  return sorted(laws)


def _relative(url: str) -> str:
  return url.removeprefix(SITE)


def _refresh(crawl: Crawl, url: str, folder: Path) -> tuple[bytes, bool]:
  """The newest capture of `url` and whether it changed since the last run."""
  capture = folder / "sources" / f"{_relative(url)}.html"
  sidecar = capture.with_suffix(".json")
  headers = {"User-Agent": AGENT}
  if capture.is_file() and sidecar.is_file():
    known = json.loads(sidecar.read_text(encoding="utf-8"))
    headers |= {
      key: value
      for key, value in (
        ("If-None-Match", known.get("etag")),
        ("If-Modified-Since", known.get("modified")),
      )
      if value
    }
  time.sleep(crawl.interval)
  response = crawl.client.get(url, headers=headers, follow_redirects=True)
  if response.status_code == httpx.codes.NOT_MODIFIED:
    return capture.read_bytes(), False
  response.raise_for_status()
  save_bytes(capture, response.content)
  state = {"etag": response.headers.get("etag"), "modified": response.headers.get("last-modified")}
  save_bytes(sidecar, json.dumps(state).encode())
  return response.content, True


def _newest(crawl: Crawl, url: str, folder: Path) -> tuple[str, bytes, bool]:
  """Follow a law page to its consolidated version, when Planalto links one."""
  payload, changed = _refresh(crawl, url, folder)
  versions = [link for link in links(decode_html(payload), url) if CONSOLIDATED.search(link)]
  if not versions:
    return url, payload, changed
  target = versions[0]
  newest, newer = _refresh(crawl, target, folder)
  return target, newest, changed or newer


def fetch_planalto(
  data_dir: Path, crawl: Crawl, statutes: Sequence[str], laws: Sequence[str]
) -> dict[str, int]:
  """Refresh every law and rewrite the articles of those that changed; returns counts."""
  if not statutes:
    raise ValueError("the statute reference bundle is empty")
  folder = data_dir / PLANALTO_DIR
  counts = {"changed": 0, "unchanged": 0, "failed": 0, "articles": 0, "dropped": 0}
  for url in laws:
    out = folder / "articles" / f"{_relative(url)}.jsonl"
    try:
      target, payload, changed = _newest(crawl, url, folder)
      if not changed and out.is_file():
        counts["unchanged"] += 1
        continue
      found = dict(articles(FederalLaw(_relative(url), _relative(target)), payload))
      if not found or any("�" in body for body in found.values()):
        raise ValueError("missing or undecodable article text")
    except (httpx.HTTPError, ValueError) as error:
      counts["failed"] += 1
      print(f"planalto: {url}: {error}")
      continue
    clean = set(without_leaks(tuple(found.values()), statutes))
    rows = [
      json.dumps(
        {"license": PUBLIC_DOMAIN, "url": target, "origin_ref": ref, "text": body},
        ensure_ascii=False,
      )
      + "\n"
      for ref, body in found.items()
      if body in clean
    ]
    save_bytes(out, "".join(rows).encode())
    counts["changed"] += 1
    counts["articles"] += len(rows)
    counts["dropped"] += len(found) - len(rows)
  return counts


def load_planalto(data_dir: Path) -> tuple[str, ...]:
  """Saved article texts, in path order; empty until fetched."""
  folder = data_dir / PLANALTO_DIR / "articles"
  return tuple(
    str(json.loads(line)["text"])
    for path in sorted(folder.rglob("*.jsonl"))
    for line in path.read_text(encoding="utf-8").splitlines()
    if line
  )
