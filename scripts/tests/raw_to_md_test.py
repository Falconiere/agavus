"""Tests for converting fetched JSON and HTML into Markdown."""

import json
from pathlib import Path

from raw_to_md import convert_tree, html_to_md, readable_statute

_HTML = """<html><body>
<p>Lei n
o
10.406</p>
<p>Art. 1 Esta Lei regula.</p>
<a href="/ccivil_03/leis/l8069.htm">Estatuto</a>
</body></html>
"""

_PAGE = {
  "items": [
    {
      "id": 9,
      "tipoDocumento": "Decisão",
      "nomeOrgao": "1ª Vara de Família",
      "numeroprocessocommascara": "0701234-56.2024.8.07.0001",
      "data_disponibilizacao": "2024-05-10",
      "link": "https://pje.example/doc/9",
      "texto": "Narra a autora. DECIDO. Ante o exposto, julgo. Art. 2 aplica-se.",
    },
    {
      "id": 9,
      "tipoDocumento": "Decisão",
      "texto": "duplicata",
      "link": "https://pje.example/doc/9",
    },
  ]
}


def test_html_keeps_links_and_articles() -> None:
  text = html_to_md(_HTML, "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406.htm")
  assert "[Estatuto](https://www.planalto.gov.br/ccivil_03/leis/l8069.htm)" in text
  assert "## Art. 1" in text
  assert "nº 10.406" in text


def test_statute_reflow_is_stable() -> None:
  once = readable_statute("Art. 2 Considera-se a guarda compartilhada do filho.")
  assert once == readable_statute(once)


def test_djen_json_becomes_one_markdown_per_id(tmp_path: Path) -> None:
  page = tmp_path / "page.json"
  page.write_text(json.dumps(_PAGE), encoding="utf-8")
  again = tmp_path / "page-2.json"
  again.write_text(json.dumps(_PAGE), encoding="utf-8")
  assert convert_tree(tmp_path) == len(("page.json", "page-2.json"))
  written = list(tmp_path.glob("*.md"))
  assert [path.name for path in written] == ["9.md"]
  body = written[0].read_text(encoding="utf-8")
  assert "[abrir no PJe](https://pje.example/doc/9)" in body
  assert "[0701234-56.2024.8.07.0001](https://pje.example/doc/9)" in body
  assert "DECIDO." in body
  assert not page.exists()
  assert not again.exists()
