# Classification pipeline

Agavus has two classifiers in production. Their output labels describe family law (F),
domestic violence (M) and matters outside scope (O). Both feed the same
[expected-loss rule](three-way-decisions.md).

| | Document command | Questions and Gradio Space |
| --- | --- | --- |
| Input | Text passed to `classify` | Question passed to `-p` or the demo |
| Representation | Normalized words, word bigrams, character trigrams, scoped negation, decision verbs, citations and citation–verb pairs | Pinned multilingual E5 query embeddings, centered and scaled |
| Estimator | Class-balanced multinomial logistic regression | Linear classifier fitted on authored question examples |
| Calibration | Temperature chosen on the separate calibration split | Temperature chosen on out-of-fold scores with scenario variants kept together |
| Explanation | Area probabilities, losses and largest term contributions | Area probabilities and losses |

The document training set uses resolved labeled texts and authored seeds. Case-derived
texts require an identifiable case number in their provenance. TPU and court metadata
help assign labels. The document vectorizer accepts text only. Vocabulary, IDF and
model weights are fitted on the training partition. Duplicate texts and records from
the same case stay together across splits. [Split details](../evals.md#splits-and-leakage)

For example, a reference to a statute and a decision verb in the same document generate
features for both and their combination. The classifier transforms these features with
the stored vocabulary, applies its weights and calibrated temperature, then compares
the expected loss of each action. The question router instead encodes the whole question.
A missing encoder causes a question to defer.

The [private learned-classifier comparison](learned-classifier.md) has its own datasets,
model weights and evaluation. Its area and statement-relation heads are experimental.
