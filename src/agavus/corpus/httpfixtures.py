"""Shared synthetic HTTP fixtures for corpus tests (no network, no real data)."""

from collections.abc import Callable
from datetime import date

import httpx

from agavus.corpus.records import CaseRecord, LabelSource

Handler = Callable[[httpx.Request], httpx.Response]


def sample_case(
  process: str = "0701234-56.2024.8.07.0001",
  orgao: str = "Vara Cível",
  assuntos: tuple[str, ...] = (),
) -> CaseRecord:
  """One synthetic case record for tests across domains."""
  return CaseRecord(
    id=process,
    tribunal="TJDFT",
    orgao_julgador=orgao,
    classe_tpu="123",
    assuntos_tpu=assuntos,
    data_ajuizamento=date(2024, 1, 2),
    movimentos=(),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="raw/x.json",
  )


DATAJUD_HIT: dict[str, object] = {
  "_source": {
    "numeroProcesso": "0701234-56.2024.8.07.0001",
    "tribunal": "TJDFT",
    "classe": {"codigo": "12345", "nome": "Procedimento Comum"},
    "orgaoJulgador": {"nome": "1ª Vara de Família"},
    "assuntos": [{"codigo": "9999", "nome": "Guarda"}],
    "dataAjuizamento": "2024-05-10T10:00:00.000Z",
    "movimentos": [{"codigo": "1", "nome": "Conclusos", "dataHora": "2024-05-11T09:00:00Z"}],
  }
}


def json_handler(body: dict[str, object]) -> Handler:
  """Handler returning a fixed JSON 200 response."""

  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=body)

  return handler


def text_handler(text: str) -> Handler:
  """Handler returning a fixed text 200 response."""

  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=text)

  return handler


def mock_client(handler: Handler) -> httpx.Client:
  """httpx client wired to a mock transport."""
  return httpx.Client(transport=httpx.MockTransport(handler))
