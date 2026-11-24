# Three-way decisions — accept / reject / defer

**Role**: the decision layer (Yao's decision-theoretic rough sets). Consumes
calibrated probabilities and a loss matrix; never forces a guess.
Future module: `decide/rule`. Design doc §7.

## Expected loss (the core rule)

$$
R(a \mid x) = \sum_{s} \lambda(a \mid s) \cdot P(s \mid x)
\qquad
a^{*} = \arg\min_a R(a \mid x)
$$

$\lambda(a \mid s)$ = cost of taking action $a$ when the true state is $s$.
Actions: accept-as-F/M/P, reject (out of scope), defer (clarify / human).

## Worked example

$P(M) = 0.55,\; P(F) = 0.35,\; P(P) = 0.05,\; P(O) = 0.05$, with the design
doc's loss matrix:

| Action | Computation | Expected loss |
| ------------- | ------------------------------------ | ------------- |
| Accept as F | $10 \cdot 0.55 + 4 \cdot 0.05 + 8 \cdot 0.05$ | 6.10 |
| Accept as M | $6 \cdot 0.35 + 3 \cdot 0.05 + 8 \cdot 0.05$ | 2.65 |
| Accept as P | $4 \cdot 0.35 + 3 \cdot 0.55 + 8 \cdot 0.05$ | 3.45 |
| Reject | $8 \cdot 0.55 + 4 \cdot 0.35 + 4 \cdot 0.05$ | 6.00 |
| **Defer** | flat cost 1 | **1.00** ✅ |

M is the most likely class — and we still **defer**. Probability alone
doesn't decide; **cost does**. With $P(M) = 0.97$ (others 0.01), accept-as-M
costs $0.17 < 1$ → accept.

## Thresholds α and β (the rule pre-solved)

Binary reduction: in-domain probability $p$. Compare accept vs defer:

$$
\lambda_{PP}\, p + \lambda_{PN}(1-p) = \lambda_{BP}\, p + \lambda_{BN}(1-p)
$$

Solving for $p$:

$$
\alpha = \frac{\lambda_{PN} - \lambda_{BN}}
{(\lambda_{PN} - \lambda_{BN}) + (\lambda_{BP} - \lambda_{PP})}
\qquad
\beta = \frac{\lambda_{BN} - \lambda_{NN}}
{(\lambda_{BN} - \lambda_{NN}) + (\lambda_{NP} - \lambda_{BP})}
$$

Subscripts: first letter = action (P accept, B defer, N reject), second =
state (P in-domain, N not).

**Maria da Penha numbers**: wrong-acceptance $\lambda_{PN} \approx 5$,
deferral $= 1$, correct $= 0$, wrong-rejection $\lambda_{NP} = 8$:

$$
\alpha = \frac{5-1}{(5-1)+(1-0)} = 0.8
\qquad
\beta = \frac{1-0}{(1-0)+(8-1)} = 0.125
$$

| Domain | α (accept ≥) | β (reject ≤) | Why |
| -------------- | ------------ | ------------ | ------------------------------------- |
| Família | ~0.86 | ~0.25 | M→F misroute is catastrophic (loss 10) |
| Maria da Penha | ~0.80 | **~0.12** | never silently drop a possible case |
| Penal | ~0.80 | ~0.25 | balanced |

**Reading**: α rises when false acceptance is expensive; β falls when false
rejection is expensive.

## Properties

- Applied at **two gates**: routing (classify) and answering
  (answer / abstain / escalate).
- Tunable **accuracy/coverage dial**: raise α → fewer, near-certain answers.
  Report both numbers (design doc §8).
- Boundary cases produce clarifying questions or human-review items — never
  forced labels. This is how real M↔F overlap is handled.
- Loss values are placeholders to calibrate on held-out TJDFT data.