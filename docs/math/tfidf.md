# TF-IDF — text as vectors

**Role**: representation. Turns a document into a numeric vector so the rest
of the math can operate on it. Future module: `classify/features`.

## Formula

$$
\text{tfidf}(t, d) = \text{count}(t, d) \cdot \log\frac{N}{\text{df}(t)}
$$

| Symbol | Meaning |
| ------------------ | ------------------------------------------------ |
| $\text{count}(t,d)$ | occurrences of term $t$ in document $d$ (local importance) |
| $N$ | total documents in the corpus |
| $\text{df}(t)$ | number of documents containing $t$ (global rarity) |

## Worked example

Corpus: $N = 1000$ TJDFT decisions.

| Term | df | $\log(N/\text{df})$ | count in doc $d$ | weight |
| ----------- | ---- | ------------------- | ---------------- | ------ |
| processo | 950 | 0.05 | 5 | 0.26 |
| guarda | 120 | 2.12 | 2 | 4.24 |
| protetiva | 20 | 3.91 | 3 | **11.74** |

"processo" is everywhere → nearly zero weight. "protetiva" is rare and
repeated in this document → dominates the vector. Document $d$ is now a point
in term-space, and documents about protective measures cluster together.

## Intuition

A word's value is **how much it sets this document apart**. Frequent terms
carry no distinguishing information; rare, repeated terms carry a lot. The
"knowledge" is in the corpus statistics, not in the formula — the same code
works for any language.

## Practical variants (implementation choices, not new math)

- **Smoothed idf**: $\log(1 + N/\text{df})$ avoids division blow-ups.
- **Sublinear tf**: $1 + \log(\text{count})$ dampens very frequent terms.
- **Char n-grams**: features like `prot`, `rotet` capture Portuguese
  morphology (protetiva/protetivas/protetivo) without a dictionary.
  In agavus these are **word-internal only** (never across word boundaries —
  cross-word grams proved to be function-word noise).
- **Word bigrams**: keep "medida protetiva", "violência doméstica" as units.
- Stopwords ("de", "da", "que") need no explicit list: high df kills them.

## Limitations

- Bag of symbols: **word order and negation are lost** ("nega a protetiva" ≈
  "concede a protetiva").
- Synonyms are different symbols ("alimentos" vs "pensão alimentícia") — the
  lexical gap that tier-4 embeddings later address.

See also: [cosine similarity](cosine-similarity.md) for comparing these
vectors; [BM25](bm25.md) for the retrieval-side cousin of tf-idf.