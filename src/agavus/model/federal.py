"""Training-only articles from the consolidated federal laws published by Planalto."""

import json
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from agavus.corpus.statutes import decode_html, html_to_text, segment_articles
from agavus.model.bigram import UNKNOWN, build_vocab
from agavus.model.data import split_articles
from agavus.model.extra import without_leaks

PLANALTO = "https://www.planalto.gov.br/ccivil_03/"
PUBLIC_DOMAIN = "Public domain (Lei 9.610/1998, art. 8º, IV)"
FEDERAL_DIR = Path("lab/text/federal")
ADCT = "ATO DAS DISPOSIÇÕES CONSTITUCIONAIS TRANSITÓRIAS"
CLT = "CONSOLIDAÇÃO DAS LEIS DO TRABALHO"
# Where a law's text ends: the place and date of signing, at a line start or opening a
# sentence; an annex; or Planalto's "not the official text" footer.
END = re.compile(
  r"(?m)(?:^|(?<=[.;,])\s+)\s*(?:Brasília|Rio\s+de\s+Janeiro),\s*(?:em\s+)?\d{1,2}\.?º?\s+de\s"
  r"|^\s*ANEXO\s*$"
  r"|Este\s+texto\s+não\s+substitui\s+o\s+publicado"
)


@dataclass(frozen=True)
class FederalLaw:
  """An allowlisted official page and the license applying to its articles."""

  name: str
  path: str
  license: str = PUBLIC_DOMAIN

  @property
  def url(self) -> str:
    """Official source used in each citation."""
    return PLANALTO + self.path

  @property
  def stem(self) -> str:
    """Stable filename shared by the source capture and its extracted articles."""
    return Path(self.path).stem


LAWS = (
  FederalLaw("CF/1988", "constituicao/constituicaocompilado.htm"),
  FederalLaw("Código Penal", "decreto-lei/del2848compilado.htm"),
  FederalLaw("CPP", "decreto-lei/del3689compilado.htm"),
  FederalLaw("CPC/2015", "_ato2015-2018/2015/lei/l13105compilada.htm"),
  FederalLaw("CLT", "decreto-lei/del5452compilado.htm"),
  FederalLaw("CTN", "leis/l5172compilado.htm"),
  FederalLaw("CDC", "leis/l8078compilado.htm"),
  FederalLaw("ECA", "leis/l8069compilado.htm"),
  FederalLaw("Lei 8.213/1991", "leis/l8213compilado.htm"),
  FederalLaw("Lei 8.212/1991", "leis/l8212compilado.htm"),
  FederalLaw("Lei 8.112/1990", "leis/l8112compilado.htm"),
  FederalLaw("Lei 9.099/1995", "leis/l9099.htm"),
  FederalLaw("Estatuto da Pessoa Idosa", "leis/2003/l10.741compilado.htm"),
  FederalLaw("LINDB", "decreto-lei/del4657compilado.htm"),
)


def articles(law: FederalLaw, payload: bytes) -> Iterator[tuple[str, str]]:
  """(origin_ref, text) per article of a captured Planalto page, struck-out text removed."""
  text = html_to_text(decode_html(payload))
  if law.name == "CLT":
    _, heading, text = text.partition(f"\n{CLT}\n")
    if not heading:
      raise ValueError("CLT heading missing")
  main, _, transitional = text.partition(f"\n{ADCT}\n")
  for prefix, section in (("", main), ("ADCT, ", transitional)):
    if not section:
      continue
    body = END.split(section, maxsplit=1)[0]
    for number, article in segment_articles(body):
      yield f"{law.url}, {prefix}art. {number}", " ".join(article.split())


def save_bytes(path: Path, payload: bytes) -> None:
  """Write whole or not at all, so an interrupted run never leaves half a file."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with TemporaryDirectory(dir=path.parent) as directory:
    staged = Path(directory) / path.name
    staged.write_bytes(payload)
    staged.replace(path)


def fetch_federal(
  data_dir: Path,
  client: httpx.Client,
  statutes: Sequence[str],
  *,
  laws: Sequence[FederalLaw] = LAWS,
  interval: float = 2.0,
) -> None:
  """Resume by law from immutable HTML captures and report the clean training text."""
  if not statutes:
    raise ValueError("the statute reference bundle is empty")
  vocab = build_vocab(split_articles(statutes).train)
  folder = data_dir / FEDERAL_DIR
  total_chars = total_unknown = total_bytes = total_kept = total_dropped = 0
  for law in laws:
    source = folder / "sources" / f"{law.stem}.html"
    if source.is_file():
      payload = source.read_bytes()
    else:
      time.sleep(interval)
      response = client.get(law.url)
      response.raise_for_status()
      payload = response.content
    found = dict(articles(law, payload))
    if not found or any("\ufffd" in body for body in found.values()):
      raise ValueError(f"{law.url}: missing or undecodable article text")
    if not source.exists():
      save_bytes(source, payload)
    clean = set(without_leaks(tuple(found.values()), statutes))
    lines = []
    kept = chars = size = unknown = 0
    for ref, body in found.items():
      if body in clean:
        row = {"law": law.name, "license": law.license, "origin_ref": ref, "text": body}
        lines.append(json.dumps(row, ensure_ascii=False) + "\n")
        kept += 1
        chars += len(body)
        size += len(body.encode())
        unknown += vocab.encode(body).count(UNKNOWN)
    save_bytes(folder / f"{law.stem}.jsonl", "".join(lines).encode())
    dropped = len(found) - kept
    print(f"{law.name}: {kept} kept, {dropped} dropped, {chars} characters, {size} bytes")
    total_kept += kept
    total_dropped += dropped
    total_chars += chars
    total_bytes += size
    total_unknown += unknown
  print(f"total: {total_kept} kept, {total_dropped} dropped, {total_bytes} UTF-8 bytes")
  fraction = total_unknown / max(1, total_chars)
  print(f"<unk>: {total_unknown}/{total_chars} characters ({fraction:.4%})")


def load_federal(data_dir: Path, laws: Sequence[FederalLaw] = LAWS) -> tuple[str, ...]:
  """Read only a complete set of allowlisted per-law JSONL files."""
  paths = [data_dir / FEDERAL_DIR / f"{law.stem}.jsonl" for law in laws]
  if not all(path.is_file() for path in paths):
    return ()
  return tuple(
    str(json.loads(line)["text"])
    for path in paths
    for line in path.read_text(encoding="utf-8").splitlines()
    if line
  )
