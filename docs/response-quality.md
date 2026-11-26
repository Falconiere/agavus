# Answer delivery

The question router can accept an area, defer for clarification or reject a question
outside scope. If the encoder is unavailable, the engine returns a deferral. A question
that names only a recognized law, without a topic or article, can also trigger clarification.
[CLI output and exit codes](cli.md)

After an area is accepted, the engine tries the area's [cited rules](reasoning.md).
When no rule matches, it retrieves statutes first, prioritizing explicit law-and-article
references. If statute retrieval yields no hits it searches the broader local corpus.
The top excerpt must share a content token or citation with the question. This lexical check
can still admit an irrelevant article. [Retrieval measurements](evals.md#retrieval-by-meaning-2026-11-25-retrieval-v1)

An accepted excerpt includes its reference, anonymized text, up to two context references
and the legal-advice disclaimer. A rule answer includes the supporting provision and
matched facts. Before delivery the verifier checks for at least one source, the disclaimer,
recognized personal-data patterns and prohibited advice phrases. Failure returns a deferral
with empty sources. These checks cannot establish legal sufficiency or detect every identifier.
[Answer composition and verifier](../src/agavus/answer/compose.py)
