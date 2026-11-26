# 4. Score the guesses

![Probability becomes loss](illustrations/loss-surprise.webp)

Keep the actual answer `a` fixed. Giving it 90%, 50%, or 10% produces loss
about 0.11, 0.69, or 2.30. The chips represent a score, not a feeling.

## Think of being surprised by an answer

If you were almost certain of an answer and it was wrong, that is a bigger
surprise than missing a difficult question. Loss turns this intuition into a
number. Giving the observed letter 50% produces about 0.69 loss; giving it
only 10% produces about 2.30. The table below shows the same rule at other chances.

**Where the analogy stops:** the model feels no surprise. Loss is a numerical
score, not an emotion, and a low text-prediction loss isn't a truth score.

Is the model any good? We need one number. It's called **loss**, and it means
**surprise**.

```mermaid
flowchart LR
  A["Model gave the right<br/>letter a HIGH chance"] --> B["Not surprised<br/>low loss"]
  C["Model gave the right<br/>letter a LOW chance"] --> D["Very surprised<br/>high loss"]
```

**Lower is better.** Zero would mean it's always certain and right.

| Chance given to the right letter | Loss |
| --- | --- |
| 100% | 0.00 |
| 90% | 0.11 |
| 50% | 0.69 |
| 25% | 1.39 |
| 10% | 2.30 |
| 1% | 4.61 |
| 0% | infinite |

You don't need to calculate it by hand. (The formula is `-ln(chance)`; your
calculator's `ln` key does it.) The last row is why lesson 3 needed smoothing:
a single "impossible" letter would break the score.

## Scoring a whole word

Score every step, then take the average. For `guarda`, with smoothing:

| Step | Chance | Loss |
| --- | --- | --- |
| ■ → g | 22% | 1.50 |
| g → u | 30% | 1.20 |
| u → a | 30% | 1.20 |
| a → r | 25% | 1.39 |
| r → d | 30% | 1.20 |
| d → a | 30% | 1.20 |
| a → ■ | 17% | 1.79 |
| **Average** | | **1.36** |

## Is 1.36 good?

Compare it with someone who knows nothing and picks any of the 8 tokens
equally. Their loss is **2.08**. Ours is lower, so the model learned something.

On the real laws:

| | Loss |
| --- | --- |
| Knows nothing (99 tokens) | 4.595 |
| **Our bigram** | **2.305** |

Only compare losses when the tokens and the test text are the same.

## Check yourself

Model A gave the right letter 50%. Model B gave it 10%. Who has the lower
loss?

<details><summary>Answer</summary>

Model A: 0.69 against 2.30. It was less surprised.

</details>

Next: [5. Use an exam](05-use-an-exam.md)
