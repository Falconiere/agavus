# Three-way decisions — accept / reject / defer

**Role**: the decision layer (Yao's decision-theoretic rough sets). Consumes
calibrated probabilities and a loss matrix; never forces a guess.
Implementation in `decide/rule`. Design doc §7.

Domains: **F** (família), **M** (Maria da Penha / violência doméstica —
protective measures and DV criminal prosecutions), **O** (out of scope,
including general criminal law).

## Expected loss (the core rule)

$$
R(a \mid x) = \sum_{s} \lambda(a \mid s) \cdot P(s \mid x)
\qquad
a^{*} = \arg\min_a R(a \mid x)
$$

$\lambda(a \mid s)$ = cost of taking action $a$ when the true state is $s$.
Actions: accept-as-F, accept-as-M, reject (out of scope), defer
(clarify / human).

## Worked example

$P(M) = 0.55,\; P(F) = 0.40,\; P(O) = 0.05$, with the design doc's loss matrix:

| Action | Computation | Expected loss |
| ------------- | --------------------------- | ------------- |
| Accept as F | $10 \cdot 0.55 + 8 \cdot 0.05$ | 5.90 |
| Accept as M | $6 \cdot 0.40 + 8 \cdot 0.05$ | 2.80 |
| Reject | $8 \cdot 0.55 + 4 \cdot 0.40$ | 6.00 |
| **Defer** | flat cost 1 | **1.00** ✅ |

M is the most likely class — and we still **defer**. Probability alone
doesn't decide; **cost does**. With $P(M) = 0.97,\; P(F) = 0.02,\;
P(O) = 0.01$, accept-as-M costs $6 \cdot 0.02 + 8 \cdot 0.01 = 0.20 < 1$
→ accept.

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

**Maria da Penha numbers**: wrong-acceptance $\lambda_{PN} \approx 7$
(average of F→M = 6 and O→M = 8), deferral $= 1$, correct $= 0$,
wrong-rejection $\lambda_{NP} = 8$:

$$
\alpha = \frac{7-1}{(7-1)+(1-0)} \approx 0.86
\qquad
\beta = \frac{1-0}{(1-0)+(8-1)} = 0.125
$$

**Família numbers**: $\lambda_{PN} \approx 9$ (average of M→F = 10 and
O→F = 8), $\lambda_{NP} = 4$:

$$
\alpha = \frac{9-1}{(9-1)+(1-0)} \approx 0.89
\qquad
\beta = \frac{1}{1+(4-1)} = 0.25
$$

| Domain | α (accept ≥) | β (reject ≤) | Why |
| -------------- | ------------ | ------------ | ------------------------------------- |
| Família | ~0.89 | ~0.25 | M→F misroute is catastrophic (loss 10) |
| Maria da Penha | ~0.86 | **~0.12** | never silently drop a possible case |

**Reading**: α rises when false acceptance is expensive; β falls when false
rejection is expensive.

## In agavus

Implemented in `decide/losses.py` + `decide/rule.py`: matrix composed from
the area registry, safety ceiling validated in code, ties prefer defer.
`agavus classify --explain` prints the full expected-loss table.

The binary reductions illustrate the costs. Production chooses the minimum across
all actions using the full probability vector. After routing, a question can also
defer when it lacks a supporting source. [Answer checks](../response-quality.md)
