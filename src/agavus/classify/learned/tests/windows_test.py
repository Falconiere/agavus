"""Token coverage and meaning-preserving rejection of oversized pairs."""

import pytest

from agavus.classify.learned.windows import OVERLAP, document_windows, relation_window

from .fixtures import tiny_tokenizer


def test_prefix_repeats_and_overlaps_reconstruct_every_payload_token() -> None:
  tokenizer = tiny_tokenizer()
  text = "pedido negado sim " * 400
  original = tokenizer.encode(text, add_special_tokens=False).ids
  prefix = tokenizer.encode("passage: ", add_special_tokens=False).ids
  recovered: list[int] = []
  for window in document_windows(tokenizer, text):
    assert window[1 : 1 + len(prefix)] == tuple(prefix)
    payload = list(window[1 + len(prefix) : -1])
    if recovered:
      assert payload[:OVERLAP] == recovered[-OVERLAP:]
      recovered.extend(payload[OVERLAP:])
    else:
      recovered.extend(payload)
  assert recovered == original
  with pytest.raises(ValueError, match="oversized statement pair"):
    relation_window(tokenizer, "pedido " * 480, "sim")
