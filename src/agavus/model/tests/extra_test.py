"""Extra lab text: Civil Code rest, Wikipedia rows, leak guard (synthetic data)."""

import json
from pathlib import Path

import httpx

from agavus.model.extra import (
  LEAK_RUN,
  WikipediaArticle,
  civil_code_rest,
  fetch_wikipedia,
  load_wikipedia,
  save_wikipedia,
  without_leaks,
)

PAGE = (
  "## Art. 1º Toda pessoa é capaz [(Redação dada pela Lei nº 1, de 2000)](http://x).\n"
  "texto que continua. Art. 1.511. O casamento estabelece comunhão plena de vida. "
  "Art. 1.784. Aberta a sucessão, a herança transmite-se. ##"
)


def _write_page(data_dir: Path, run: str, text: str) -> None:
  folder = data_dir / "raw" / "statutes" / run
  folder.mkdir(parents=True)
  (folder / "Lei 10.406-2002.md").write_text(text, encoding="utf-8")


def test_civil_code_rest_skips_the_family_book_and_markdown(tmp_path: Path) -> None:
  _write_page(tmp_path, "2026-01-01", PAGE)
  assert civil_code_rest(tmp_path) == (
    "Art. 1º Toda pessoa é capaz (Redação dada pela Lei nº 1, de 2000). texto que continua.",
    "Art. 1.784. Aberta a sucessão, a herança transmite-se.",
  )


def test_civil_code_rest_reads_the_newest_page(tmp_path: Path) -> None:
  _write_page(tmp_path, "2026-01-01", PAGE)
  _write_page(tmp_path, "2026-02-01", "Art. 2º A personalidade civil começa.")
  assert civil_code_rest(tmp_path) == ("Art. 2º A personalidade civil começa.",)


def test_civil_code_rest_is_empty_without_a_page(tmp_path: Path) -> None:
  assert civil_code_rest(tmp_path) == ()


def _rows(request: httpx.Request) -> httpx.Response:
  offset = int(request.url.params["offset"])
  total = 5
  rows = [
    {"row": {"title": f"T{n}", "url": f"https://pt.wikipedia.org/wiki/T{n}", "text": f"texto {n}"}}
    for n in range(offset, min(offset + int(request.url.params["length"]), total))
  ]
  return httpx.Response(200, json={"rows": rows})


def test_fetch_wikipedia_pages_cuts_and_stops() -> None:
  client = httpx.Client(transport=httpx.MockTransport(_rows))
  found = fetch_wikipedia(client, articles=3, chars=5)
  assert [article.title for article in found] == ["T0", "T1", "T2"]
  assert found[0].text == "texto"
  assert len(fetch_wikipedia(client, articles=50, chars=99)) == len("12345")


def test_wikipedia_round_trip_keeps_attribution(tmp_path: Path) -> None:
  articles = (WikipediaArticle("Água", "https://pt.wikipedia.org/wiki/%C3%81gua", "A água é"),)
  path = save_wikipedia(tmp_path, articles)
  assert json.loads(path.read_text(encoding="utf-8"))["url"].endswith("%C3%81gua")
  assert load_wikipedia(tmp_path) == ("A água é",)


def test_load_wikipedia_is_empty_before_fetching(tmp_path: Path) -> None:
  assert load_wikipedia(tmp_path) == ()


def test_without_leaks_drops_long_shared_runs_only() -> None:
  protected = "x" * LEAK_RUN + " fim"
  copied = "início " + "x" * LEAK_RUN
  spaced = "início " + "  ".join(["x" * (LEAK_RUN // 2)] * 2)[: LEAK_RUN + 2]
  boilerplate = "início " + "x" * (LEAK_RUN - 1)
  assert without_leaks([copied, boilerplate, "outro"], [protected]) == (boilerplate, "outro")
  assert without_leaks([spaced], ["x" * (LEAK_RUN // 2) + " " + "x" * (LEAK_RUN // 2)]) == ()
