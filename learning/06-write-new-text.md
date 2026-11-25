# 6. Write new text

Guess a letter, then use it to guess the next one. Repeat. That's how a model
writes.

```mermaid
flowchart TD
  A["Start at ■"] --> B["Look up the chances<br/>for the current letter"]
  B --> C["Pick one at random,<br/>using those chances"]
  C --> D{"Is it ■ ?"}
  D -->|"no"| E["Write it down.<br/>It becomes the current letter"]
  E --> B
  D -->|"yes"| F["Stop"]
```

"At random, using those chances" is like a wheel with slices. `r` with 50% has
a slice twice as big as a space with 25%. Likely letters win often, not always.

## What ours writes

After reading the real laws:

> Ass cr prece denúntídana elulhe ... deiação qu ...

The pairs of letters look Portuguese, such as `qu` and `ção`. The words don't
exist, because **it remembers only one letter**.

## Two things to remember

1. **Sounding right isn't being right.** The model picks likely letters. It
   doesn't know what is true. This is why Agavus answers with quoted laws.
2. **More memory makes better text.** The next models look further back:

```mermaid
flowchart LR
  A["Bigram<br/>1 letter back"] --> B["Better tokens<br/>pieces of words"] --> C["Small neural model<br/>several tokens back"] --> D["Tiny GPT<br/>chooses what to look at"]
```

## Check yourself

Why can't our bigram reliably write the word `guarda`?

<details><summary>Answer</summary>

At `a` it can't tell whether it's at the first `a` (then `r`) or the last one
(then the end). It sees only `a`, so it has to pick by chance.

</details>

Back to the [lesson list](README.md)
