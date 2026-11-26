# Corpus privacy

Real case records, raw fetches, decision excerpts, case-derived labels and research
artifacts stay in the private data directory, outside Git. Public fixtures must be
synthetic. The public bootstrap uses bundled statute texts and authored routing examples.

Process only material available through authorized sources. Exclude sealed material and
respect portal access controls. Raw evidence may retain identifying details and needs
restricted access even when a processed excerpt has been redacted.

The text redactor recognizes CPF, CEP, email, phone numbers, labeled RG/CPF/CNPJ
identifiers and some names following party-role markers. It preserves URLs. The
residual-data detector checks CPF, email and phone patterns. Names without markers,
addresses and other identifying combinations can survive these checks. A clean scan
does not authorize publication of case-derived material.
[Redaction code](../../src/agavus/corpus/anonymize.py) · [Publishing policy](../publishing.md)

The staged-data guard rejects files under `data/`. If you configure another
`AGAVUS_DATA_DIR`, choose a location outside version control. [Storage](storage.md)
