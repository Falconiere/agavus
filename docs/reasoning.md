# Reasoning with cited rules

After routing accepts an area, the engine extracts named facts and their spans from the
question, with a negation guard. It selects the first rule in that area whose conditions
are all present. Each rule names the exact statutory reference required to support its
fixed conclusion. [Rules](../src/agavus/reason/rules.py) · [Chain construction](../src/agavus/reason/chain.py)

| Rule | Required facts | Reference in the corpus |
| --- | --- | --- |
| R-24A | Existing protective measure and breach | Lei 11.340/2006, art. 24-A |
| R-22 | Domestic violence, family context and urgency | Lei 11.340/2006, art. 22 |
| R-AP | Disqualification campaign and obstruction of contact | Lei 12.318/2010, art. 2 |

For example, a question containing both the existing-measure and breach facts can
trigger R-24A. The answer shows the rule identifier, its fixed conclusion, the spans
it matched and the stored text of art. 24-A. If the cited provision is missing, the
engine defers. When no rule matches, the engine searches for an excerpt instead.
The [answer checks](response-quality.md) apply to both paths.
