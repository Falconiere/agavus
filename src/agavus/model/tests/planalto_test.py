"""Planalto crawl on synthetic pages: discovery, newest consolidated text, conditional refresh."""

import json
from pathlib import Path

import httpx

from agavus.corpus.httpfixtures import mock_client
from agavus.model.extra import LEAK_RUN
from agavus.model.federal import PUBLIC_DOMAIN
from agavus.model.planalto import (
  AGENT,
  FIXED,
  INDEXES,
  PLANALTO_DIR,
  SITE,
  Crawl,
  discover,
  fetch_planalto,
  links,
  load_planalto,
)

COPIED = "x" * LEAK_RUN
ORDINARY = SITE + INDEXES[0]
YEAR = SITE + "_ato2023-2026/2024/lei/_leis2024.htm"
AMENDED = SITE + "leis/l8069.htm"
COMPILED = SITE + "leis/l8069compilado.htm"
IN_PLACE = SITE + "_ato2023-2026/2024/lei/l14900.htm"
MISSING = SITE + "leis/l1.htm"
PAGES = {
  ORDINARY: '<a href="../_Ato2023-2026/2024/Lei/_leis2024.htm">2024</a>'
  '<a href="L8069.htm">ECA</a><a href="../MPV/Principal.htm">MPV</a>',
  YEAR: '<a href="L14900.htm">14.900</a><a href="../Msg/Vep/VEP-1-24.htm">veto</a>'
  '<a href="../../../leis/_Lei-Ordinaria.htm">back</a>',
  AMENDED: '<a href="L8069compilado.htm">Texto compilado</a><p>Art. 1º Texto original.</p>',
  COMPILED: f"<p>Art. 1º Texto <strike>antigo</strike> novo.</p><p>Art. 2º {COPIED}.</p>",
  IN_PLACE: "<p>Art. 1º Lei atual.</p><p>Brasília, 1º de março.</p><p>Assinatura</p>",
  **{SITE + path: "<p>Sem leis.</p>" for path in INDEXES[1:]},
}


def serve(seen: list[httpx.Request]) -> httpx.Client:
  def handler(request: httpx.Request) -> httpx.Response:
    seen.append(request)
    url = str(request.url)
    if url not in PAGES:
      return httpx.Response(404)
    etag = f'"{len(PAGES[url])}"'
    if request.headers.get("if-none-match") == etag:
      return httpx.Response(304)
    return httpx.Response(200, headers={"ETag": etag}, content=PAGES[url].encode("cp1252"))

  return mock_client(handler)


def test_links_are_absolute_lowercase_planalto_urls() -> None:
  html = (
    '<a href="../Leis/L8069.htm#art1">a</a><a href="https://example.org/ccivil">b</a>'
    "<a href='https://portaldalegislacao.presidencia.gov.br/ccivil_03/Decreto-Lei/Del2848.htm'>c</a>"
  )
  assert list(links(html, SITE + "codigos/quadro_cod.htm")) == [
    SITE + "leis/l8069.htm",
    SITE + "decreto-lei/del2848.htm",
  ]


def test_discover_walks_indexes_and_saved_portal_listings(tmp_path: Path) -> None:
  listing = tmp_path / "decretos-leis.html"
  listing.write_text('<a href="../../ccivil_03/decreto-lei/Del2848.htm">CP</a>', encoding="utf-8")
  requests: list[httpx.Request] = []
  with serve(requests) as client:
    laws = discover(Crawl(client, interval=0), [listing])
  assert laws == sorted(
    {AMENDED, IN_PLACE, SITE + "decreto-lei/del2848.htm", *(SITE + path for path in FIXED)}
  )
  fetched = [str(request.url) for request in requests]
  assert fetched.count(ORDINARY) == 1
  assert YEAR in fetched
  assert all(request.headers["user-agent"] == AGENT for request in requests)


def test_fetch_keeps_newest_text_and_refreshes_only_changes(tmp_path: Path) -> None:
  requests: list[httpx.Request] = []
  laws = [AMENDED, IN_PLACE, MISSING]
  with serve(requests) as client:
    crawl = Crawl(client, interval=0)
    first = fetch_planalto(tmp_path, crawl, (COPIED, "abc"), laws)
    assert first == {"changed": 2, "unchanged": 0, "failed": 1, "articles": 2, "dropped": 1}
    assert load_planalto(tmp_path) == ("Art. 1º Lei atual.", "Art. 1º Texto novo.")
    row = json.loads(
      (tmp_path / PLANALTO_DIR / "articles" / "leis" / "l8069.htm.jsonl").read_text(
        encoding="utf-8"
      )
    )
    assert row == {
      "license": PUBLIC_DOMAIN,
      "url": COMPILED,
      "origin_ref": f"{COMPILED}, art. 1",
      "text": "Art. 1º Texto novo.",
    }
    requests.clear()
    again = fetch_planalto(tmp_path, crawl, (COPIED, "abc"), laws)
  assert again == {"changed": 0, "unchanged": 2, "failed": 1, "articles": 0, "dropped": 0}
  assert all("if-none-match" in r.headers for r in requests if str(r.url) != MISSING)
  assert load_planalto(tmp_path) == ("Art. 1º Lei atual.", "Art. 1º Texto novo.")
