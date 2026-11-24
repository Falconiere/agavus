"""F/M subject terms for ingestion. The area is the scope; there is no start date."""

SUBJECT_PHRASES: tuple[str, ...] = (
  "violência doméstica",
  "medida protetiva",
  "maria da penha",
  "feminicídio",
  "alienação parental",
  "divórcio",
  "guarda",
  "alimentos",
  "união estável",
  "11340",
  "12318",
  "13058",
  "14994",
  "14550",
)

DATAJUD_FIELDS: tuple[str, ...] = ("orgaoJulgador.nome", "assuntos.nome", "classe.nome")

# Diary search is a single string. Statute numbers stay as "Lei N", not bare digits,
# so a process number that happens to contain 11340 is not a hit.
DJEN_TERMS: tuple[str, ...] = (
  "violência doméstica",
  "medida protetiva",
  "maria da penha",
  "feminicídio",
  "alienação parental",
  "divórcio",
  "guarda",
  "alimentos",
  "união estável",
  "Lei 11.340",
  "Lei 12.318",
  "Lei 13.058",
  "Lei 14.994",
  "Lei 14.550",
)
