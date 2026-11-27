"""Agathe: windows, attention never looks ahead, learning (synthetic text)."""

import pytest
import torch

from agavus.model.agathe import AgatheConfig, attend, start, stream, windows
from agavus.model.bigram import BOUNDARY, CharVocab, build_vocab
from agavus.model.training import IGNORED, pick

# Many threads only fight over work this small.
torch.set_num_threads(1)

TEXTS = ["guarda guarda", "a guarda do filho"]
CONFIG = AgatheConfig(context=8, width=8, heads=2, layers=2)
STEPS = 200


@pytest.mark.parametrize("context", [1, 2, 4, 8, 64])
@pytest.mark.parametrize("text", ["", "ab", "guarda guarda guarda guarda"])
def test_windows_score_every_token_once(text: str, context: int) -> None:
  vocab = build_vocab([text])
  inputs, answers = windows([text], vocab, context)
  scored = [int(token) for token in answers.flatten() if token != IGNORED]
  assert scored == vocab.encode(text)[1:]
  assert inputs.shape == answers.shape


def test_windows_inputs_sit_one_token_before_answers() -> None:
  vocab = build_vocab(TEXTS)
  inputs, answers = windows(TEXTS, vocab, 4)
  ids = vocab.encode(TEXTS[0])
  assert inputs[0].tolist() == ids[:4]
  assert answers[0].tolist() == ids[1:5]


def test_stream_joins_texts_with_one_boundary() -> None:
  vocab = build_vocab(TEXTS)
  first, second = (vocab.encode(text) for text in TEXTS)
  ids = stream(TEXTS, vocab)
  assert ids.tolist() == first + second[1:]
  assert ids.dtype == torch.uint8


def test_attention_never_looks_ahead() -> None:
  model = start(build_vocab(TEXTS), CONFIG)
  for looks in model.looks("guarda"):
    ahead = torch.ones_like(looks[0], dtype=torch.bool).triu(1)
    assert torch.all(looks[:, ahead] == 0)
    assert torch.allclose(looks.sum(-1), torch.ones(looks.shape[:2]))


def test_changing_a_later_token_leaves_earlier_scores_alone() -> None:
  vocab = build_vocab(TEXTS)
  model = start(vocab, CONFIG)
  first = torch.tensor([vocab.encode("guarda")])
  second = torch.tensor([vocab.encode("guardo")])
  with torch.no_grad():
    assert torch.allclose(model.scores(first)[0, :5], model.scores(second)[0, :5])


def test_looks_cover_the_start_and_each_letter() -> None:
  model = start(build_vocab(TEXTS), CONFIG)
  assert [layer.shape for layer in model.looks("guarda")] == [(2, 7, 7)] * 2
  assert model.looks("guarda guarda")[0].shape == (2, 8, 8)


def test_lab_parameter_counts() -> None:
  vocab = CharVocab(tuple(chr(i) for i in range(97)))
  sizes = (start(vocab, AgatheConfig()).size(), start(vocab, AgatheConfig(layers=5)).size())
  assert (vocab.size, *sizes) == (99, 822_627, 1_019_875)


def test_seeded_continuation_starts_after_the_prefix_and_stops_at_boundary() -> None:
  vocab = build_vocab(["ab"])
  model = start(vocab, AgatheConfig(width=vocab.size, heads=1, layers=1))
  boundary, a, b, _ = vocab.encode("ab")
  with torch.no_grad():
    for weight in model.parameters():
      weight.zero_()
    model.tokens.copy_(torch.eye(vocab.size))
    for current, following in ((boundary, a), (a, b), (b, boundary)):
      model.out_weights[current, following] = 100
  assert model.sample(seed=7, max_chars=200) == "ab"
  assert list(model.continuation("a", seed=7, max_chars=1)) == [b]
  assert list(model.continuation("b", seed=7, max_chars=200)) == []
  assert model.sample(seed=7, max_chars=1) == "a"


def test_same_context_suffix_has_the_same_seeded_continuation() -> None:
  model = start(build_vocab(TEXTS), CONFIG)
  suffix = "a guarda"
  full = list(model.continuation("um começo diferente " + suffix, seed=7, max_chars=20))
  assert full and full == list(model.continuation(suffix, seed=7, max_chars=20))


def test_fused_attention_matches_the_shown_attention() -> None:
  model = start(build_vocab(["abcdef ghij"]), AgatheConfig(context=16, width=12, heads=3, layers=1))
  vectors = torch.randn(2, 16, 12, generator=torch.Generator().manual_seed(0))
  with torch.no_grad():
    shown, looks = attend(model.layers[0], vectors, 3, show=True)
    fused, none = attend(model.layers[0], vectors, 3, show=False)
  assert looks is not None
  assert none is None
  assert torch.allclose(shown, fused, atol=1e-6)


def test_cached_continuation_matches_reading_the_whole_window_each_time() -> None:
  model = start(build_vocab(TEXTS), AgatheConfig(context=8, width=12, heads=3, layers=2))
  prefix, seed, chars = "a guarda", 3, 30
  with torch.no_grad():
    model.out_bias[BOUNDARY] = -1e9  # never stop early, so the window must slide
  generator = torch.Generator().manual_seed(seed)
  ids = model.vocab.encode(prefix)[:-1]
  expected: list[int] = []
  with torch.no_grad():
    while len(expected) < chars:
      chosen = pick(model.scores(torch.tensor([ids[-8:]]))[0, -1], generator)
      if chosen == BOUNDARY:
        break
      ids.append(chosen)
      expected.append(chosen)
  assert len(expected) == chars
  assert list(model.continuation(prefix, seed=seed, max_chars=chars)) == expected
