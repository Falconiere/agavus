# Math notes

One document per formula in the agavus core. The math is **language-agnostic**:
it operates on symbols and counts; Portuguese-specific concerns live in
preprocessing (tokenization, n-grams, citation regexes), one layer below.

| Formula | Role in the pipeline | Future module | Design doc |
| ----------------------- | ---------------------------- | ---------------- | ---------- |
| [TF-IDF](tfidf.md) | represent text as vectors | `classify/features` | §6 |
| [Cosine similarity](cosine-similarity.md) | compare vectors | `retrieve` | §6 |
| [Logistic regression](logistic-regression.md) | scores → class probabilities | `classify/model` | §6 |
| [Calibration](calibration.md) | make probabilities honest | `classify/model` | §6 |
| [BM25](bm25.md) | rank documents for a query | `retrieve/search` | §6 |
| [Three-way decisions](three-way-decisions.md) | accept / reject / defer | `decide/rule` | §7 |

End-to-end view: **[classification-pipeline.md](classification-pipeline.md)**
— how records become feature vectors become predictions.

Reading order: TF-IDF → logistic regression → calibration → BM25 →
three-way decisions. Cosine can be read any time after TF-IDF.

One line each: **TF-IDF represents, logreg votes, calibration tells the
truth, BM25 finds, expected loss decides.**