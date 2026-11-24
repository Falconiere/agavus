"""Polite HTTP fetching: rate limit, retry with backoff, raw preservation.

DataJud is frequently overloaded (429/5xx, slow queries) — patience is a
feature. Every successful payload is preserved verbatim under data/raw/.
"""

import json
import time
from dataclasses import dataclass
from datetime import date
from typing import cast

import httpx

from agavus.corpus.storage import CorpusStorage

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
HTTP_ERROR_MIN = 400


class FetchError(RuntimeError):
  """A fetch failed after exhausting retries."""


@dataclass(frozen=True)
class FetchPolicy:
  """Patience knobs: throttle, retries, backoff, timeout.

  Defaults are deliberately conservative (0.5 req/s, Retry-After honored,
  backoff capped at 60s) — we never want to be blocked by a public API.
  """

  min_interval: float = 2.0
  max_retries: int = 4
  backoff_base: float = 3.0
  max_backoff: float = 60.0
  timeout: float = 90.0


@dataclass(frozen=True)
class JsonRequest:
  """One fetch request: method, url, optional headers/payload, raw name."""

  method: str
  url: str
  headers: dict[str, str] | None = None
  payload: dict[str, object] | None = None
  name: str = "payload"


class Fetcher:
  """Rate-limited HTTP client that preserves every successful payload."""

  def __init__(
    self,
    storage: CorpusStorage,
    source: str,
    *,
    client: httpx.Client | None = None,
    policy: FetchPolicy | None = None,
  ) -> None:
    self._storage = storage
    self._source = source
    self._policy = policy if policy is not None else FetchPolicy()
    self._client = client if client is not None else httpx.Client(timeout=self._policy.timeout)
    self._last_call = 0.0
    self._seq = 0
    self.last_raw_ref = ""

  def _throttle(self) -> None:
    wait = self._policy.min_interval - (time.monotonic() - self._last_call)
    if wait > 0:
      time.sleep(wait)
    self._last_call = time.monotonic()

  def _retry_wait(self, attempt: int, response: httpx.Response | None) -> float:
    """Honor Retry-After when present; otherwise capped exponential backoff."""
    if response is not None:
      retry_after = response.headers.get("Retry-After", "")
      if retry_after.isdigit():
        return float(retry_after)
    return min(self._policy.backoff_base**attempt, self._policy.max_backoff)

  def fetch(self, request: JsonRequest) -> bytes:
    """Fetch with retries; preserve raw payload + query meta; return bytes."""
    policy = self._policy
    for attempt in range(policy.max_retries + 1):
      self._throttle()
      try:
        response = self._client.request(
          request.method, request.url, headers=request.headers, json=request.payload
        )
      except httpx.HTTPError as exc:
        if attempt >= policy.max_retries:
          raise FetchError(f"{request.url}: {exc}") from exc
        time.sleep(self._retry_wait(attempt, None))
        continue
      if response.status_code in RETRY_STATUS and attempt < policy.max_retries:
        time.sleep(self._retry_wait(attempt, response))
        continue
      if response.status_code >= HTTP_ERROR_MIN:
        raise FetchError(f"{request.url}: HTTP {response.status_code}")
      self._seq += 1
      self.last_raw_ref = self._storage.save_raw(
        self._source,
        date.today(),
        f"{request.name}-{self._seq:04d}.json",
        response.content,
        {"url": request.url, "method": request.method, "payload": request.payload},
      )
      return response.content
    raise FetchError(f"{request.url}: retries exhausted")

  def fetch_json(self, request: JsonRequest) -> dict[str, object]:
    """Fetch and parse a JSON object response."""
    return cast(dict[str, object], json.loads(self.fetch(request)))
