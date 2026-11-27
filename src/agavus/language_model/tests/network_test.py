"""Tiny network: hand-worked lesson values, true gradients, learning."""

from dataclasses import replace

import numpy as np
import pytest

from agavus.language_model.network import (
  CLUES,
  REFUSED,
  Network,
  forward,
  gradients,
  loss,
  squash,
  start,
  train,
  train_single_neuron,
)

HIDDEN = 8
NUDGE = 1e-6
SOLVED = 0.1


def test_lesson_hidden_layer_by_hand() -> None:
  by_hand = Network(
    hidden_weights=np.ones((2, 2)),
    hidden_bias=np.array([0.0, -1.0]),
    out_weights=np.array([1.0, -2.0]),
    out_bias=0.0,
  )
  hidden = forward(by_hand, CLUES).hidden
  assert (hidden @ by_hand.out_weights).tolist() == REFUSED.tolist()


def test_lesson_neuron_values() -> None:
  assert squash(np.array([-2.0, 2.0])) == pytest.approx([0.119, 0.881], abs=1e-3)


def test_backpropagation_matches_measured_slopes() -> None:
  network = start(HIDDEN, seed=1)
  network = replace(network, hidden_bias=np.full(HIDDEN, 0.05))
  slope = gradients(network, CLUES, REFUSED)

  def loss_of(changed: Network) -> float:
    return loss(forward(changed, CLUES).guess, REFUSED)

  for index in np.ndindex(network.hidden_weights.shape):
    nudge = np.zeros_like(network.hidden_weights)
    nudge[index] = NUDGE
    up = loss_of(replace(network, hidden_weights=network.hidden_weights + nudge))
    down = loss_of(replace(network, hidden_weights=network.hidden_weights - nudge))
    assert slope.hidden_weights[index] == pytest.approx((up - down) / (2 * NUDGE), abs=1e-6)
  up = loss_of(replace(network, out_bias=network.out_bias + NUDGE))
  down = loss_of(replace(network, out_bias=network.out_bias - NUDGE))
  assert slope.out_bias == pytest.approx((up - down) / (2 * NUDGE), abs=1e-6)


def test_network_learns_what_one_neuron_cannot() -> None:
  trained, losses = train(start(HIDDEN, seed=0), CLUES, REFUSED, steps=2000, rate=0.5)
  assert losses[-1] < SOLVED < losses[0]
  assert np.round(forward(trained, CLUES).guess).tolist() == REFUSED.tolist()
  alone = train_single_neuron(CLUES, REFUSED, steps=2000, rate=0.5)
  assert alone == pytest.approx([0.5] * 4)


def test_training_is_deterministic() -> None:
  runs = [train(start(HIDDEN, seed=4), CLUES, REFUSED, steps=20, rate=0.5)[1] for _ in range(2)]
  assert runs[0] == runs[1]
