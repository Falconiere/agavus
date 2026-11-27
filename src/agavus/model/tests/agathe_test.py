"""Agathe: windows, causality, RoPE, GQA, tied output, cache, learning (synthetic text)."""

import math
from dataclasses import replace

import pytest
import torch
from torch import Tensor

from agavus.model.agathe import (
  END_ID,
  EVAL_BATCH,
  EVAL_SCORES,
  AgatheConfig,
  AgatheModel,
  check_config,
  framed,
  hidden_size,
  placed,
  rotary,
  rotate,
  start,
  stream,
  train,
  windows,
)
from agavus.model.data import Split
from agavus.model.tokenizer import encode, train_tokenizer
from agavus.model.training import IGNORED, pick

# Many threads only fight over work this small.
torch.set_num_threads(1)

TEXTS = [
  "Art. 22. Constatada a prática de violência doméstica contra a mulher, o juiz decide.",
  "Art. 19. As medidas protetivas de urgência poderão ser concedidas pelo juiz.",
  "A guarda do filho será compartilhada entre os pais.",
] * 10
TOKENIZER = train_tokenizer(TEXTS, 300)
CONFIG = AgatheConfig(context=16, width=32, heads=4, kv_heads=2, layers=2)


@pytest.mark.parametrize("context", [1, 2, 4, 16, 64])
@pytest.mark.parametrize("text", ["", "ab", TEXTS[0]])
def test_windows_score_every_token_once(text: str, context: int) -> None:
  inputs, answers = windows([text], TOKENIZER, context)
  scored = [int(token) for token in answers.flatten() if token != IGNORED]
  assert scored == framed(TOKENIZER, text)[1:]
  assert inputs.shape == answers.shape


def test_stream_joins_texts_with_one_end_token_in_two_bytes() -> None:
  first, second = (framed(TOKENIZER, text) for text in TEXTS[:2])
  ids = stream(TEXTS[:2], TOKENIZER)
  assert ids.long().tolist() == first + second[1:]
  assert ids.dtype == torch.uint16


@pytest.mark.parametrize(
  ("width", "heads", "kv_heads"), [(30, 4, 2), (32, 4, 3), (12, 4, 2), (0, 4, 2)]
)
def test_config_refuses_sizes_heads_or_rope_cannot_split(
  width: int, heads: int, kv_heads: int
) -> None:
  with pytest.raises(ValueError):
    check_config(AgatheConfig(width=width, heads=heads, kv_heads=kv_heads))


def test_changing_a_later_token_leaves_earlier_scores_alone() -> None:
  model = start(TOKENIZER, CONFIG)
  first = torch.tensor([framed(TOKENIZER, TEXTS[0])[:12]])
  second = first.clone()
  second[0, 8:] = (second[0, 8:] + 5) % TOKENIZER.get_vocab_size()
  with torch.no_grad():
    assert torch.allclose(model.scores(first)[0, :8], model.scores(second)[0, :8])
    assert not torch.allclose(model.scores(first)[0, 8:], model.scores(second)[0, 8:])


def test_attention_never_looks_ahead_and_matches_the_fused_scores() -> None:
  model = start(TOKENIZER, CONFIG)
  shown = model.looks(TEXTS[1])
  assert [layer.shape[:2] for layer in shown] == [(CONFIG.heads, CONFIG.context)] * 2
  for looks in shown:
    ahead = torch.ones_like(looks[0], dtype=torch.bool).triu(1)
    assert torch.all(looks[:, ahead] == 0)
    assert torch.allclose(looks.sum(-1), torch.ones(looks.shape[:2]))


def test_rope_dot_products_depend_only_on_the_distance() -> None:
  size, shift = 8, 37
  query, key = torch.randn(2, 1, 1, 1, size, generator=torch.Generator().manual_seed(1))

  def dot(at_query: int, at_key: int) -> Tensor:
    cos, sin = rotary(size, torch.tensor([at_query, at_key]), 10000.0)
    turned = rotate(torch.cat((query, key), dim=2), cos, sin)
    return (turned[..., 0, :] * turned[..., 1, :]).sum()

  assert torch.allclose(dot(5, 2), dot(5 + shift, 2 + shift), atol=1e-5)
  assert not torch.allclose(dot(5, 2), dot(5, 4), atol=1e-5)


def _shared_heads(model: AgatheModel, heads: int) -> AgatheModel:
  """The same model with each key/value head copied for every query head in its group."""
  group = heads // model.config.kv_heads
  size = model.config.width // heads

  def copied(weights: Tensor) -> Tensor:
    split = weights.view(model.config.width, model.config.kv_heads, size)
    return split.repeat_interleave(group, dim=1).reshape(model.config.width, heads * size)

  layers = tuple(
    replace(layer, key=copied(layer.key), value=copied(layer.value)) for layer in model.layers
  )
  return replace(model, config=replace(model.config, kv_heads=heads), layers=layers)


def test_grouped_heads_equal_full_heads_with_copied_keys_and_values() -> None:
  model = start(TOKENIZER, CONFIG)
  full = _shared_heads(model, CONFIG.heads)
  assert full.layers[0].key.shape == (CONFIG.width, CONFIG.width)
  assert model.layers[0].key.shape == (CONFIG.width, CONFIG.width // 2)
  ids = torch.tensor([framed(TOKENIZER, TEXTS[2])[:16]])
  with torch.no_grad():
    assert torch.allclose(model.scores(ids), full.scores(ids), atol=1e-6)


def test_tied_output_and_parameter_count_formula() -> None:
  vocab = TOKENIZER.get_vocab_size()
  for config in (CONFIG, replace(CONFIG, kv_heads=4, layers=3), AgatheConfig()):
    model = start(TOKENIZER, config)
    width, hidden = config.width, hidden_size(config.width)
    shared = config.kv_heads * width // config.heads
    per_layer = 2 * width + 2 * width * width + 2 * width * shared + 3 * width * hidden
    assert model.size() == vocab * width + config.layers * per_layer + width
    assert sum(weight.shape[-1] == vocab for weight in model.parameters()) == 0
  assert hidden_size(512) == 21 * 64


def test_seeded_continuation_starts_after_the_prefix_and_stops_at_the_end_token() -> None:
  model = start(TOKENIZER, CONFIG)
  # Layers add nothing; every token points the same way, the end token ten times further.
  with torch.no_grad():
    for layer in model.layers:
      layer.mix.zero_()
      layer.down.zero_()
    model.tokens.zero_()
    model.tokens[:, 0] = 1.0
    model.tokens[END_ID, 0] = 10.0
    model.gain.fill_(10.0)
  assert list(model.continuation("a guarda", seed=7, max_tokens=50)) == []
  assert model.sample(seed=7, max_chars=50) == ""
  with torch.no_grad():
    model.tokens[END_ID, 0] = -10.0
  assert len(list(model.continuation("a guarda", seed=7, max_tokens=5))) == len("abcde")


def _recomputed(model: AgatheModel, prefix: str, seed: int, count: int) -> list[int]:
  """Seeded tokens read from the whole recent window each step, without a cache."""
  generator = torch.Generator().manual_seed(seed)
  ids = framed(TOKENIZER, prefix)[:-1]
  written: list[int] = []
  with torch.no_grad():
    while len(written) < count:
      chosen = pick(model.scores(torch.tensor([ids[-model.config.context :]]))[0, -1], generator)
      if chosen == END_ID:
        break
      ids.append(chosen)
      written.append(chosen)
  return written


def test_cached_continuation_matches_reading_the_whole_window_until_it_slides() -> None:
  model = start(TOKENIZER, CONFIG)
  with torch.no_grad():
    model.tokens[END_ID] = -model.tokens[1:].sum(0)  # never stop early
  prefix, seed = "a guarda", 3
  fill = CONFIG.context - len(encode(TOKENIZER, prefix))
  assert list(model.continuation(prefix, seed=seed, max_tokens=fill)) == _recomputed(
    model, prefix, seed, fill
  )
  long = list(model.continuation(prefix, seed=seed, max_tokens=3 * CONFIG.context))
  assert len(long) == 3 * CONFIG.context
  assert long[:fill] == _recomputed(model, prefix, seed, fill)


def test_training_reduces_loss_and_reports_bits_per_character() -> None:
  config = replace(CONFIG, batch=8, rate=1e-2, check_every=10)
  split = Split(tuple(TEXTS), tuple(TEXTS[:3]))
  blank = start(TOKENIZER, config)
  model, history = train(split, TOKENIZER, config, steps=30)
  assert len(history) == 30 // config.check_every
  assert model.loss(TEXTS[:3]) < blank.loss(TEXTS[:3]) - 1
  bits = model.bits_per_char(TEXTS[:3])
  assert 0 < bits < blank.bits_per_char(TEXTS[:3]) and math.isfinite(bits)


def test_placed_copies_the_weights_into_a_model_for_another_config() -> None:
  model = start(TOKENIZER, CONFIG)
  moved = placed(model, replace(CONFIG, seed=1, bf16=True))
  assert moved.config.bf16 and moved is not model
  assert all(torch.equal(a, b) for a, b in zip(moved.parameters(), model.parameters(), strict=True))


def test_eval_rows_keep_scores_within_the_memory_budget() -> None:
  small = start(TOKENIZER, AgatheConfig(context=16, width=16, heads=4, kv_heads=2, layers=1))
  assert small.eval_rows() == EVAL_BATCH
  wide = start(TOKENIZER, AgatheConfig(context=8192, width=16, heads=4, kv_heads=2, layers=1))
  assert wide.eval_rows() < EVAL_BATCH
  assert wide.eval_rows() * wide.config.context * TOKENIZER.get_vocab_size() <= EVAL_SCORES
  assert wide.eval_rows() >= 1
