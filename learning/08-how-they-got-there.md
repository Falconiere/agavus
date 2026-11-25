# 8. How people got to the formula

Nobody wrote it in one go. Each piece fixed a problem left by another.

```mermaid
flowchart TD
  P1["Problem: what usually<br/>comes next?"] --> S1["Markov, 1913:<br/>count and divide"]
  S1 --> P2["Problem: never seen<br/>means impossible?"]
  P2 --> S2["Laplace, about 1774:<br/>add 1"]
  S2 --> P3["Problem: how do we<br/>measure surprise?"]
  P3 --> S3["Shannon, 1948:<br/>count the halvings"]
```

(Laplace came first in time; Markov's idea comes first in ours.)

## Markov: count and divide

Andrey Markov, a Russian mathematician, took 20,000 letters of a poem and
counted vowels and consonants by hand.

| Previous letter | Chance that the next is a vowel |
| --- | --- |
| vowel | about 13% |
| consonant | about 66% |

So **the letter before changes what comes next**. Our bigram is the same idea
with every letter: `2 / 4`.

## Laplace: add 1

Pierre-Simon Laplace asked: if the sun rose every day so far, what is the
chance it rises tomorrow? Saying 100% is too confident: we can't have seen
everything. His answer was to pretend we saw each outcome once more.

With two outcomes (rises or not), after 9 sunrises in 9 days:
`(9 + 1) / (9 + 2) = 10/11`, about 91%. Our formula does this with 8 outcomes
instead of 2: `(2 + 1) / (4 + 8)`.

## Shannon: measure surprise

Claude Shannon wanted a number for surprise. What should it do?

1. Something certain: surprise **0**.
2. Something rarer: **more** surprise.
3. Two separate surprises: they should **add up**.

Try coins. Each head cuts the chance in half:

| Heads in a row | Chance | Surprise we want |
| --- | --- | --- |
| 0 | 1 | 0 |
| 1 | 1/2 | 1 |
| 2 | 1/4 | 2 |
| 3 | 1/8 | 3 |

So **surprise = how many times you halve 1 to get the chance**. All three rules
work. That operation already had a name: the **logarithm** (base 2).

```mermaid
flowchart LR
  A["1"] -->|"half"| B["1/2"] -->|"half"| C["1/4"] -->|"half"| D["1/8"]
```

## Why do we write ln, then?

`ln` is the same measure in another unit, like centimeters and inches.
**One halving = 0.69.**

| Chance | Halvings | × 0.69 | = our loss |
| --- | --- | --- | --- |
| 1/2 | 1 | 0.69 | 0.69 |
| 1/4 | 2 | 2 × 0.693 | 1.39 |
| 1/8 | 3 | 3 × 0.693 | 2.08 |

The last one is our "knows nothing" score, with 8 tokens. Mathematicians like
`ln` because it makes later formulas simpler.

## Put it together

```mermaid
flowchart LR
  A["Count and divide<br/>Markov"] --> D["Chance of the<br/>right letter"]
  B["Add 1<br/>Laplace"] --> D
  D --> C["Halvings<br/>Shannon"] --> E["Average = loss"]
```

## Check yourself

The chance is 1/16. How many halvings is that?

<details><summary>Answer</summary>

4: 1 → 1/2 → 1/4 → 1/8 → 1/16.

</details>

Back to the [lesson list](README.md)
