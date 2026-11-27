"""Tiny neural network by hand: learning/11-neural-networks.md in code.

Two clues in, a few hidden neurons, one chance out. Backpropagation is written
step by step, without a library, because seeing it is the point.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
# Clues: has "não", has "negou". Answer: was the request refused?
CLUES: FloatArray = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
REFUSED: FloatArray = np.array([0.0, 1.0, 1.0, 0.0])
TEXTS = ("concedeu", "não concedeu", "negou", "não negou")
TINY = 1e-12


@dataclass(frozen=True)
class Network:
  """Weights and biases of one hidden layer and one output neuron."""

  hidden_weights: FloatArray
  hidden_bias: FloatArray
  out_weights: FloatArray
  out_bias: float


@dataclass(frozen=True)
class Forward:
  """Everything computed on the way from clues to guess."""

  sums: FloatArray
  hidden: FloatArray
  guess: FloatArray


def squash(value: FloatArray) -> FloatArray:
  """Any number into 0..1 (sigmoid)."""
  return np.asarray(1 / (1 + np.exp(-value)), dtype=np.float64)


def start(hidden: int, seed: int) -> Network:
  """Small random weights: equal ones would all learn the same thing."""
  rng = np.random.default_rng(seed)
  return Network(
    hidden_weights=rng.normal(size=(2, hidden)),
    hidden_bias=np.zeros(hidden),
    out_weights=rng.normal(size=hidden),
    out_bias=0.0,
  )


def forward(network: Network, clues: FloatArray) -> Forward:
  """Multiply, add, drop negatives; multiply, add, squash."""
  sums = clues @ network.hidden_weights + network.hidden_bias
  hidden = np.maximum(sums, 0.0)
  guess = squash(hidden @ network.out_weights + network.out_bias)
  return Forward(sums, hidden, guess)


def loss(guess: FloatArray, answers: FloatArray) -> float:
  """Mean surprise at the right answers (lesson 4)."""
  chance = np.where(answers == 1.0, guess, 1 - guess)
  return float(-np.log(chance + TINY).mean())


def gradients(network: Network, clues: FloatArray, answers: FloatArray) -> Network:
  """Backpropagation: each weight's share of the mistake, from end to start."""
  passed = forward(network, clues)
  mistake = (passed.guess - answers) / len(answers)
  blame = np.outer(mistake, network.out_weights) * (passed.sums > 0)
  return Network(
    hidden_weights=clues.T @ blame,
    hidden_bias=blame.sum(axis=0),
    out_weights=passed.hidden.T @ mistake,
    out_bias=float(mistake.sum()),
  )


def step(network: Network, slope: Network, rate: float) -> Network:
  """Move every weight a little downhill."""
  return Network(
    hidden_weights=network.hidden_weights - rate * slope.hidden_weights,
    hidden_bias=network.hidden_bias - rate * slope.hidden_bias,
    out_weights=network.out_weights - rate * slope.out_weights,
    out_bias=network.out_bias - rate * slope.out_bias,
  )


def train(
  network: Network, clues: FloatArray, answers: FloatArray, *, steps: int, rate: float
) -> tuple[Network, list[float]]:
  """Guess, measure, find the slope, step; returns every loss on the way."""
  losses = [loss(forward(network, clues).guess, answers)]
  for _ in range(steps):
    network = step(network, gradients(network, clues, answers), rate)
    losses.append(loss(forward(network, clues).guess, answers))
  return network, losses


def train_single_neuron(
  clues: FloatArray, answers: FloatArray, *, steps: int, rate: float
) -> FloatArray:
  """Guesses of one neuron without a hidden layer, for comparison."""
  weights = np.zeros(clues.shape[1])
  bias = 0.0
  for _ in range(steps):
    mistake = (squash(clues @ weights + bias) - answers) / len(answers)
    weights = weights - rate * (clues.T @ mistake)
    bias -= rate * float(mistake.sum())
  return squash(clues @ weights + bias)
