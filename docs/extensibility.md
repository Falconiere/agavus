# Extending the area registry

Production covers two TJDFT routing areas and an out-of-scope state. Each frozen
`AreaSpec` has a code, display name, status and tribunal tuple. The registry defines
`planned`, `collecting`, `shadow` and `active` status values. Family law and domestic
violence are marked collecting and both participate in routing. The current routing
selection excludes the out-of-scope entry and does not filter by lifecycle status.
[Area registry](../src/agavus/taxonomy/areas.py)

The classifier, rule selection and cost matrix use area codes from the registry.
The corpus stores records and derived labels in a shared SQLite database. The statute
mapping, organ patterns, rule pack and authored evaluation sets supply the area-specific
content. [Labeling](corpus/labeling.md) · [Rules](reasoning.md)

## Requirements for an extension

A new tribunal or domain requires a design change before production ingestion. Define the
sources and provenance, TPU and organ mappings, terminology, statutory references,
area-scoped rules and evaluation examples. Validate the composed losses, accepted
precision, coverage and confusion with M before enabling the area. The global safety
ceilings remain 10 for misrouting M to F and 8 for rejecting M.

The design calls for shadow evaluation before activation. A status field alone does
not enforce that boundary. Private comparative research retains the tribunal and
deciding body on each record. See [the production scope](DESIGN.md#2-domain-scope).
