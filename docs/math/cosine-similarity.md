# Cosine similarity — comparing vectors

**Role**: measures how aligned two TF-IDF vectors are; a retrieval and
clustering signal. Future module: `retrieve`.

## Formula

$$
\cos(x, y) = \frac{x \cdot y}{\lVert x \rVert \, \lVert y \rVert}
= \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2}\,\sqrt{\sum_i y_i^2}}
$$

Range for non-negative TF-IDF vectors: $[0, 1]$.
$1$ = same direction (same topic mix), $0$ = no shared terms.

## Worked example

Two documents, two terms (weights from the [TF-IDF](tfidf.md) example):

| Doc | protetiva | alimentos |
| --- | --------- | --------- |
| $x$ | 3.9 | 0 |
| $y$ | 3.9 | 2.12 |

$$
\cos(x, y) = \frac{3.9 \cdot 3.9 + 0}{3.9 \cdot \sqrt{3.9^2 + 2.12^2}}
= \frac{15.21}{3.9 \cdot 4.44} \approx 0.88
$$

Very similar, but not identical — $y$ also talks about alimentos.

## Intuition

The **angle** between documents, ignoring magnitude. A long acórdão and a
short decisão about the same topic point the same way even though their
vector lengths differ. That is exactly what we want for topical similarity.

## Limitations

- Lexical only: paraphrases with disjoint vocabulary score ~0.
- Sensitive to representation choices (n-grams, normalization) — fix those
  upstream, not here.

See also: [BM25](bm25.md), the query–document scorer used for retrieval
ranking (cosine is the document–document companion).