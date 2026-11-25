"""Convert fetched JSON and HTML under a directory into readable Markdown.

Anchors, images, and DJEN document URLs stay as Markdown links. Each DJEN
communication is written once, named by its id. The source file is removed
after a successful write. Run: uv run python scripts/raw_to_md.py [data/raw]
"""

import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from agavus.corpus.jsonutil import as_dict, as_list, as_str
from agavus.corpus.statutes import STATUTES

WIDTH = 88
PARAGRAPH = 420
_LINK = re.compile(r"\[[^\[\]]*\]\([^)]+\)")
_ORDINAL_N = re.compile(r"\bn\s+o\s+(?=\d)", re.IGNORECASE)
_ORDINAL_D = re.compile(r"(\d+)\s+o\b")
_ARTICLE = re.compile(r"\s+(Art\.\s*\d+)")
_STRUCT = re.compile(
  r"\s+(§\s*\d+|Parágrafo único\.?|(?:CAPÍTULO|TÍTULO|SEÇÃO|SUBSEÇÃO|LIVRO)\b[^\n]{0,80})"
)
_DASH = chr(0x2013)
_INCISO = re.compile(rf"\s+((?:I{{1,3}}|IV|VI{{0,3}}|IX|XI{{0,3}}|X{{1,3}})\s+[-{_DASH}]\s+)")
_SECTION = re.compile(
  r"\s+((?:É o relatório|DECIDO|Ante o exposto|DISPOSITIVO)\b\.?)",
  re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<![A-ZÁÉÍÓÚÂÊÔÃÕÇ]\.)(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ\"“])")


def statute_base(name: str) -> str:
  """Map `Lei 11.340-2006.html` to the Planalto page used as link base."""
  stem = name.removesuffix(".html")
  head, _, year = stem.rpartition("-")
  key = f"{head}/{year}" if head else stem
  for statute in STATUTES:
    if statute.name == key:
      return statute.url
  return "https://www.planalto.gov.br/"


def _shield(text: str) -> tuple[str, list[str]]:
  found: list[str] = []

  def repl(match: re.Match[str]) -> str:
    found.append(re.sub(r"\s+", " ", match.group(0)))
    return f"\x00{len(found) - 1}\x00"

  return _LINK.sub(repl, text), found


def _restore(text: str, found: list[str]) -> str:
  for index, link in enumerate(found):
    text = text.replace(f"\x00{index}\x00", link)
  return text


def _unwrap(text: str) -> str:
  flat = re.sub(r"\s*\n\s*", " ", text.replace("\r", ""))
  flat = re.sub(r"[ \t]{2,}", " ", flat).strip()
  flat = _ORDINAL_N.sub("nº ", flat)
  return _ORDINAL_D.sub(r"\1º", flat)


def _reflow(paragraph: str) -> str:
  words = paragraph.split()
  lines: list[str] = []
  current: list[str] = []
  width = 0
  for word in words:
    extra = len(word) if not current else len(word) + 1
    if current and width + extra > WIDTH and not word.startswith("]("):
      lines.append(" ".join(current))
      current = [word]
      width = len(word)
    else:
      current.append(word)
      width += extra
  if current:
    lines.append(" ".join(current))
  return "\n".join(lines)


def readable_statute(text: str) -> str:
  """One block per article, inciso, and heading, wrapped to a reading width."""
  shielded, links = _shield(text)
  if "\n## Art." in shielded:
    parts = [part.strip() for part in re.split(r"\n\n+", shielded) if part.strip()]
    body = "\n\n".join(_reflow(part) for part in parts)
    return _restore(body, links).rstrip() + "\n"
  flat = _unwrap(shielded)
  flat = _STRUCT.sub(r"\n\n\1", flat)
  flat = _ARTICLE.sub(r"\n\n## \1", flat)
  flat = _INCISO.sub(r"\n\n- \1", flat)
  parts = [part.strip() for part in re.split(r"\n\n+", flat) if part.strip()]
  return _restore("\n\n".join(_reflow(part) for part in parts), links).rstrip() + "\n"


def _pack(block: str) -> list[str]:
  sentences = _SENTENCE.split(block)
  packed: list[str] = []
  buf = ""
  for sentence in sentences:
    if buf and len(buf) + len(sentence) > PARAGRAPH:
      packed.append(buf.strip())
      buf = sentence
    else:
      buf = f"{buf} {sentence}".strip()
  if buf:
    packed.append(buf.strip())
  return packed


def readable_decision(text: str) -> str:
  """Break a decision into short paragraphs and wrap the lines."""
  shielded, links = _shield(text)
  flat = _SECTION.sub(r"\n\n\1", _unwrap(shielded))
  blocks = [part.strip() for part in re.split(r"\n\n+", flat) if part.strip()]
  packed = [part for block in blocks for part in _pack(block)]
  return _restore("\n\n".join(_reflow(part) for part in packed), links).rstrip() + "\n"


def html_to_md(html: str, base: str) -> str:
  """Visible text with anchors and images kept as Markdown links."""
  soup = BeautifulSoup(html, "html.parser")
  for tag in soup.find_all(["script", "style", "noscript"]):
    tag.decompose()
  for tag in soup.find_all("link"):
    href = tag.get("href")
    if isinstance(href, str) and tag.get("rel") != ["stylesheet"]:
      tag.replace_with(f"[link]({urljoin(base, href)})")
    else:
      tag.decompose()
  for tag in soup.find_all("a"):
    href = tag.get("href")
    label = tag.get_text(" ", strip=True)
    if isinstance(href, str) and href.strip():
      shown = (label or href).replace("[", "").replace("]", "")
      tag.replace_with(f"[{shown}]({urljoin(base, href.strip())})")
    else:
      tag.replace_with(label)
  for tag in soup.find_all("img"):
    src = tag.get("src")
    if isinstance(src, str) and src.strip():
      alt = tag.get("alt")
      label = alt.strip() if isinstance(alt, str) and alt.strip() else "imagem"
      tag.replace_with(f"[{label}]({urljoin(base, src.strip())})")
  return readable_statute(soup.get_text("\n", strip=True))


def _body_md(text: str) -> str:
  stripped = text.strip()
  if "<a " in stripped.lower() or "<p" in stripped.lower() or "<div" in stripped.lower():
    return readable_decision(html_to_md(stripped, "https://comunica.pje.jus.br/"))
  return readable_decision(stripped)


def djen_item_md(item: dict[str, object]) -> str:
  """One communication: a short header plus the text, with the PJe URL kept."""
  kind = as_str(item.get("tipoDocumento")) or as_str(item.get("tipoComunicacao")) or "Comunicação"
  organ = as_str(item.get("nomeOrgao"))
  number = as_str(item.get("numeroprocessocommascara")) or as_str(item.get("numero_processo"))
  published = as_str(item.get("data_disponibilizacao"))
  link = as_str(item.get("link"))
  lines = [f"# {kind}", ""]
  if organ:
    lines.append(f"- Órgão: {organ}")
  if number and link:
    lines.append(f"- Processo: [{number}]({link})")
  elif number:
    lines.append(f"- Processo: {number}")
  if published:
    lines.append(f"- Data: {published}")
  if link:
    lines.append("- Documento: [abrir no PJe](" + link + ")")
  lines.extend(["", _body_md(as_str(item.get("texto")))])
  return "\n".join(lines).rstrip() + "\n"


def item_ident(item: dict[str, object]) -> str:
  """Stable file name for one communication."""
  ident = item.get("id")
  if isinstance(ident, int):
    return str(ident)
  if isinstance(ident, str) and ident.strip():
    return ident.strip()
  return as_str(item.get("hash"))


def _read_html(path: Path) -> str:
  raw = path.read_bytes()
  for encoding in ("utf-8", "cp1252", "latin-1"):
    try:
      return raw.decode(encoding)
    except UnicodeError:
      continue
  return raw.decode("latin-1", errors="replace")


def convert_html(path: Path) -> Path:
  """Replace one HTML file with Markdown next to it."""
  target = path.with_suffix(".md")
  target.write_text(html_to_md(_read_html(path), statute_base(path.name)), encoding="utf-8")
  path.unlink()
  return target


def _write_item(path: Path, item: dict[str, object]) -> Path | None:
  ident = item_ident(item)
  if not ident:
    return None
  target = path.parent / f"{ident}.md"
  if not target.exists():
    target.write_text(djen_item_md(item), encoding="utf-8")
  return target


def _other_json_md(path: Path, payload: dict[str, object]) -> Path:
  lines = [f"# {path.stem}", ""]
  for key in ("url", "link"):
    value = as_str(payload.get(key))
    if value.startswith("http"):
      lines.append(f"- [{key}]({value})")
  hits = as_dict(payload.get("hits")).get("hits")
  if isinstance(hits, list):
    for hit in hits:
      number = as_str(as_dict(as_dict(hit).get("_source")).get("numeroProcesso"))
      lines.append(f"- Processo: {number}" if number else "- hit")
  if payload.get("error"):
    lines.extend(["", "Consulta recusada pelo índice (pool de busca cheio)."])
  target = path.with_suffix(".md")
  target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
  return target


def convert_json(path: Path) -> int:
  """Replace one JSON file with Markdown. DJEN items are deduped by id."""
  payload = json.loads(path.read_text(encoding="utf-8"))
  written = 0
  if isinstance(payload, dict) and isinstance(payload.get("items"), list):
    for raw in as_list(payload.get("items")):
      if _write_item(path, as_dict(raw)) is not None:
        written += 1
  elif isinstance(payload, dict):
    _other_json_md(path, payload)
    written = 1
  else:
    target = path.with_suffix(".md")
    preview = path.read_text(encoding="utf-8")[:2000]
    target.write_text(f"```json\n{preview}\n```\n", encoding="utf-8")
    written = 1
  path.unlink()
  meta = Path(str(path) + ".meta.json")
  if meta.exists():
    meta.unlink()
  return written


def convert_tree(root: Path) -> int:
  """Convert every JSON and HTML file under root. Returns the source count."""
  sources = [
    path
    for path in root.rglob("*")
    if path.is_file() and path.suffix in {".json", ".html"} and not path.name.endswith(".meta.json")
  ]
  for path in sources:
    if path.suffix == ".html":
      convert_html(path)
    else:
      convert_json(path)
  return len(sources)


def main(argv: Sequence[str] | None = None) -> int:
  """Convert a raw directory. Defaults to data/raw next to the working directory."""
  args = list(sys.argv[1:] if argv is None else argv)
  root = Path(args[0]) if args else Path("data/raw")
  if not root.is_dir():
    print(f"not a directory: {root}")
    return 1
  print(f"converted {convert_tree(root)} files under {root}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
