# 17. More text

The tiny GPT (lesson 16) memorized: loss 0.87 on the articles it studied, 1.18
on the exam. The usual cure is more text. So we tried three kinds.

```sh
uv run python -m agavus.lab fetch-wikipedia   # once
uv run python -m agavus.lab gpt civil         # or: gpt wikipedia, gpt civil wikipedia
```

## Where the text came from

| Text | Size | Style | License |
| --- | --- | --- | --- |
| our statute articles | 148 KB | law | public domain |
| **rest of the Civil Code** (everything outside the family book) | 524 KB | law | public domain |
| **Wikipedia**, opening of 350 articles | 485 KB | encyclopedia | CC BY-SA 4.0 |

Laws aren't protected by copyright in Brazil (Lei 9.610/1998, art. 8º, IV).
Wikipedia lets anyone reuse its text if they credit it, so we keep each
article's link. Nothing we train is published.

## Keeping the exam fair

Three rules, so the new scores compare with the old ones:

1. Extra text only goes to the **study** side. Validation and the exam are the
   same statute articles as before.
2. The **alphabet** stays the statute one (99 tokens). A letter it doesn't
   know (0.45% of Wikipedia) becomes "unknown".
3. **No leaks.** Any extra text that shares 100 letters in a row with one of our
   347 articles is dropped. One Civil Code article was (art. 4º).

Same model, same 6,000 training steps for every run.

## The results

| Study text | Exam | Studied articles | Gap |
| --- | --- | --- | --- |
| statutes only | **1.181** | 0.87 | 0.31 |
| + Civil Code | **1.174** | 1.06 | 0.11 |
| + Wikipedia | 1.289 | 1.21 | 0.08 |
| + both | 1.276 | 1.21 | 0.06 |

"Studied articles" is the loss on the statute articles it trained on. The
**gap** is how much better it does on text it saw than on text it didn't:
the memorizing.

## What it means

**More text cures memorizing.** The gap fell from 0.31 to 0.11, and 0.06 with
both. With more to read, it can't learn each article by heart.

**But the score barely moved.** 6,000 steps of 32 windows of 64 letters is about
12 million letters read. Over our ~120 KB of study articles, that's about 100
readings of each letter; over the Civil Code mix (~640 KB), about 19. The
model now sees each text fewer times, and validation was **still improving at the last step** of the Civil
Code run:

| Step | 4,000 | 5,000 | 6,000 |
| --- | --- | --- | --- |
| Validation, + Civil Code | 1.258 | 1.219 | 1.191 |

It ran out of **time**, not text. With more text, you also need more training.
That is the job for a GPU.

**Similar text helps more than different text.** Wikipedia made the exam
worse, even though it cut memorizing too. Its steps are spent on astronomy and
monkeys, not on "§ 1º" and "parágrafo único". Real LLMs read everything
because they have the time to; a small budget is best spent on text like the
exam.

## Check yourself

1. Why must the extra text never contain an exam article?
2. Gap 0.31 before, 0.11 after. What changed in how the model learns?
3. Validation is still falling at the last step. What would you try next?

<details><summary>Answers</summary>

1. The model would have studied the answers; the exam would measure memory, not
   learning.
2. It memorizes less: with more text, it sees each article fewer times.
3. Train longer (more steps), which is cheaper on a GPU.

</details>

Back to the [lesson list](README.md)
