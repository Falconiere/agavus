# Calibration — making probabilities honest

**Role**: corrects classifier confidence so that "0.9" empirically means 90%
correct. Mandatory before the decision layer. Future module: `classify/model`.

## The problem

A classifier can rank well and still lie about confidence: among predictions
of ~0.9, only 75% may be correct. Thresholds and expected-loss decisions
spend real costs against these numbers, so they must be trustworthy.

**Reliability check**: bin held-out predictions by confidence; plot mean
confidence vs observed accuracy. Perfect calibration = the diagonal.

## Platt scaling (the formula)

Fit a sigmoid on the model's score $z$, using a held-out set:

$$
P_{\text{cal}} = \sigma(a \cdot z + b) = \frac{1}{1 + e^{-(a z + b)}}
$$

$a, b$ are learned by minimizing log-loss on the held-out data (2 parameters
— cheap, works with small gold sets).

## Worked example

Held-out reliability before calibration:

| Predicted confidence | Observed accuracy |
| -------------------- | ----------------- |
| 0.90 | 0.75 |
| 0.60 | 0.55 |

After fitting Platt: raw 0.90 → **0.78**, raw 0.60 → **0.55**. Now a
threshold of $\alpha \approx 0.86$ (Maria da Penha accept) genuinely filters
for ~86%-correct decisions, as the loss matrix assumes.

## Isotonic alternative

A monotone step function fitted to the reliability curve — more flexible than
Platt, but needs more data to avoid overfitting. With our small gold set,
**Platt first**; isotonic later if data grows.

## Rules

- Fit on **held-out** data only — never the training set (it would look
  perfectly calibrated and lie).
- Re-calibrate whenever the model or corpus changes.
- Report calibration quality (e.g., expected calibration error) alongside
  F1 in evaluation (design doc §8).