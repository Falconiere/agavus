# TF-IDF document features

The document classifier fits a vocabulary using document frequencies in its training
partition. By default it keeps terms found in at least two documents and at most
20,000 features. Inference reuses this vocabulary. [Vectorizer](../../src/agavus/classify/features.py)

For a retained term occurring `c` times in a document, with document frequency `df`
among `N` training documents, the unnormalized weight is

```text
(1 + ln(c)) × (1 + ln((1 + N) / (1 + df)))
```

Absent or unknown features contribute zero. The vectorizer divides each nonzero row
by its L2 norm. For example, in nine training documents a term present in four has
IDF `1 + ln(10/5) ≈ 1.693`. Three occurrences give a weight of about 3.553 before
normalization.

Feature tokens include normalized words, adjacent word pairs, within-word character
trigrams, negation marks, decision verbs, citations and citation–verb pairs. The
resulting vector feeds the [document classifier](classification-pipeline.md).
