# 9. Where does the article number come from?

Question: *"Qual artigo da Lei Maria da Penha trata das medidas protetivas?"*

A language model doesn't look the number up. It guesses likely next tokens.

| | Where the number comes from | Can it be wrong? |
| --- | --- | --- |
| Our bigram | chance of a digit; ignores the question | almost always |
| A large LLM alone | patterns left in its weights by training | yes, and it sounds sure |
| LLM + search (RAG) | the law's text, found first | yes, if the search is wrong |

The LLM alone is an exam **from memory**. RAG is an **open book**.

## What Agavus does

There is no LLM in it. Each article is stored with its number as a label:

```mermaid
flowchart LR
  Q["Question"] --> C["Classify:<br/>Maria da Penha"]
  C --> S["Search<br/>the laws"]
  S --> A["Best article:<br/>its label is the number"]
  A --> R["Copy the text<br/>and the citation"]
```

## It can still be wrong

When this lesson was written, Agavus answered **art. 10**. The right answer is
**art. 22** (through 24), which the search had placed third. The search
matches words, and art. 10 also says "medida protetiva". It didn't know that
art. 22 is the one that *defines* them.

```mermaid
flowchart LR
  A["Find the right article<br/>search"] --> B["Read it correctly<br/>LLM"] --> C["Check the citation<br/>verifier"] --> D["Good answer"]
```

If the first box fails, the others can't save the answer. **A real source
doesn't guarantee the right answer.**

## Check yourself

An LLM reads the wrong article and explains it perfectly. Which box failed?

<details><summary>Answer</summary>

The first: search.

</details>

Back to the [lesson list](README.md)
