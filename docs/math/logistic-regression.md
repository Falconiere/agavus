# Logistic regression — evidence voting → probabilities

**Role**: domain classifier. Converts a TF-IDF vector into class
probabilities $P(F), P(M), P(O)$. Future module: `classify/model`.

## Formulas

Score per class (a weighted vote over features):

$$
z_k = w_k \cdot x + b_k
$$

Softmax turns scores into probabilities summing to 1:

$$
P(k \mid x) = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

Training minimizes **cross-entropy** over labeled examples:

$$
L = -\sum_{\text{examples}} \log P(\text{true class} \mid x)
$$

## Worked example

Scores for one document: $z_F = 1.0,\; z_M = 2.0,\; z_O = 0$.

| Class | $e^{z_k}$ | $P(k \mid x)$ |
| --- | ------ | ------ |
| F | 2.718 | 0.245 |
| M | 7.389 | **0.665** |
| O | 1.000 | 0.090 |
| sum | 11.107 | 1.000 |

If the true class is M, the loss is $-\ln 0.665 = 0.41$. Had the model been
confidently wrong ($P(M) = 0.05$), the loss would be $-\ln 0.05 = 3.0$ —
**confident errors are punished hardest**.

## Intuition

- Each feature votes: learned weights make "protetiva" vote strongly for M,
  "guarda" for F. Weights are inspectable — the model is auditable.
- Softmax is a soft argmax: it amplifies score differences into a probability
  distribution.
- Cross-entropy is convex in the weights → gradient descent reaches the
  **global optimum**. CPU training takes minutes, no GPU.

## In agavus

Implemented in `classify/model.py`: hand-rolled numpy full-batch gradient
descent (deterministic, convex), **class-balanced** sample weights, L2.
No sklearn/torch in the core — inspectable weights, CPU-minutes.

## Limitations

- Linear decision surface: cannot express interactions like "guarda AND
  violência" unless we add those features explicitly (bigrams help).
- Raw probabilities are often **overconfident** — mandatory next step:
  [calibration](calibration.md). The decision layer spends real costs
  against these numbers.

See also: [three-way decisions](three-way-decisions.md), which consumes the
calibrated probabilities.