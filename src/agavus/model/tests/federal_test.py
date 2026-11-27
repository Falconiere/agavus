"""Federal study text uses complete allowlisted pages and the frozen statute exam."""

import json
from pathlib import Path

import httpx
import pytest

from agavus.corpus.httpfixtures import mock_client
from agavus.corpus.statutes import STATUTES
from agavus.model.bundle import BUNDLE_FILE, Texts, federal_texts, write_bundle
from agavus.model.extra import LEAK_RUN
from agavus.model.federal import (
  ADCT,
  CLT,
  FEDERAL_DIR,
  LAWS,
  FederalLaw,
  fetch_federal,
  load_federal,
)

FIRST = FederalLaw("CF/1988", "synthetic.htm")
SECOND = FederalLaw("CLT", "other.htm")
COPIED = "x" * LEAK_RUN
PAGE = (
  "<p>Art. 1º Ação <s>old</s><strike>deleted</strike><del>gone</del>."
  '<span style="text-decoration:line-through">expired</span></p>'
  f"<p>Art. 2º Passa a vigorar: {COPIED}.</p>"
  f"<p>{ADCT}</p><p>Art. 1º Texto transitório.</p>"
  "<p>Brasília, 12 de março.</p><p>Assinatura</p>"
)
SYNTHETIC_CLT = (
  f"<p>Art. 1º Decreto.</p><p>{CLT}</p><p>Art. 1º O trabalho.</p>"
  "<p>Art. 1º-A. Mais texto.</p><p>Art. . 2. Outra regra.</p><p>ANEXO</p><p>Tabela</p>"
)


def test_federal_catalog_is_disjoint_from_exam_laws() -> None:
  assert {law.url.lower() for law in LAWS}.isdisjoint(law.url.lower() for law in STATUTES)
  assert len({law.stem for law in LAWS}) == len(LAWS)


def test_interrupted_download_resumes_and_filters_by_article(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  requests: list[str] = []
  laws = (FIRST, SECOND)
  protected = (COPIED, "abc")

  def serve(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    requests.append(url)
    if url == SECOND.url and requests.count(url) == 1:
      return httpx.Response(503)
    html = PAGE if url == FIRST.url else SYNTHETIC_CLT
    return httpx.Response(200, content=html.encode("cp1252"))

  with mock_client(serve) as client:
    with pytest.raises(httpx.HTTPStatusError):
      fetch_federal(tmp_path, client, protected, laws=laws, interval=0)
    assert not load_federal(tmp_path, laws)
    fetch_federal(tmp_path, client, protected, laws=laws, interval=0)
    first = load_federal(tmp_path, laws)
    fetch_federal(tmp_path, client, protected, laws=laws, interval=0)
    assert first == load_federal(tmp_path, laws)
  assert requests == [FIRST.url, SECOND.url, SECOND.url]
  assert first == (
    "Art. 1º Ação .",
    "Art. 1º Texto transitório.",
    "Art. 1º O trabalho.",
    "Art. 1º-A. Mais texto.",
    "Art. . 2. Outra regra.",
  )
  lines = (tmp_path / FEDERAL_DIR / "synthetic.jsonl").read_text(encoding="utf-8").splitlines()
  rows = [json.loads(line) for line in lines]
  assert rows[0]["license"] == FIRST.license
  assert [row["origin_ref"] for row in rows] == [
    FIRST.url + ", art. 1",
    FIRST.url + ", ADCT, art. 1",
  ]
  report = capsys.readouterr().out
  assert "CF/1988: 2 kept, 1 dropped" in report
  chars = "".join(first)
  unknown = sum(char not in protected[1] for char in chars)
  assert f"<unk>: {unknown}/{len(chars)} characters" in report


def test_empty_response_can_be_retried(tmp_path: Path) -> None:
  with (
    mock_client(lambda _: httpx.Response(200, text="<p>Unavailable</p>")) as client,
    pytest.raises(ValueError, match="missing or undecodable"),
  ):
    fetch_federal(tmp_path, client, ["a", "b"], laws=(FIRST,), interval=0)
  assert not list(tmp_path.rglob("*.html"))


def write_federal_inputs(data_dir: Path, base: Texts, text: str) -> Path:
  """Reference bundle plus one synthetic article per catalogued law; returns the federal folder."""
  write_bundle(data_dir / BUNDLE_FILE, base)
  folder = data_dir / FEDERAL_DIR
  folder.mkdir(parents=True)
  for law in LAWS:
    (folder / f"{law.stem}.jsonl").write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
  return folder


def test_local_federal_inputs_use_only_the_reference_bundle(tmp_path: Path) -> None:
  base = Texts(("a", "b"), {"civil": ("extra",)})
  folder = write_federal_inputs(tmp_path, base, "synthetic")
  (folder / "unexpected.jsonl").write_text("invalid", encoding="utf-8")
  merged = federal_texts(tmp_path)
  assert merged.statutes == base.statutes
  assert merged.extras == {**base.extras, "federal": ("synthetic",) * len(LAWS)}
