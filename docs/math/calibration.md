# Temperature calibration

For class logits `z`, the engine computes `softmax(z / T)`. It fits a single positive
temperature by minimizing negative log likelihood over 49 candidates from 0.2 to 5.
A larger temperature spreads probability more evenly over classes. Scaling leaves the
highest-scoring class unchanged, but it can change the minimum-loss action.
[Implementation](../../src/agavus/classify/calibrate.py) · [Decision rule](three-way-decisions.md)

For example, with logits (2, 0) the first class receives about 0.88 at T = 1 and
0.73 at T = 2. These are probabilities of a routing class. A reliability check groups
predictions by confidence and compares mean confidence to observed accuracy.

The document classifier fits vocabulary and weights on training data, fits temperature
on a separate calibration partition and reports accuracy, macro-F1 and expected calibration
error on the final holdout. The question router fits temperature using out-of-fold logits.
Each fold learns its own centering and scale, and paired variants of an authored scenario
stay in one fold. Small samples still carry substantial uncertainty.
[Evaluation and split policy](../evals.md#splits-and-leakage)
