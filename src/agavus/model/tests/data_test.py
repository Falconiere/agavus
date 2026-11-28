"""Lab data policy: statutes only, whole-article split, one-way imports."""

import ast
from pathlib import Path

import agavus
from agavus.corpus.records import LabelSource, TextDocument, TextSource
from agavus.corpus.storage import CorpusStorage
from agavus.model.bigram import build_vocab
from agavus.model.data import (
  load_articles,
  load_case_texts,
  load_statute_articles,
  split_articles,
)

PACKAGE = Path(agavus.__file__).parent
LAB_PACKAGES = {"model", "server"}


def document(name: str, source: TextSource, text: str) -> TextDocument:
  """A synthetic corpus text whose id and origin_ref are `name`."""
  return TextDocument(
    id=name,
    source=source,
    origin_ref=name,
    published_at=None,
    text=text,
    citations=(),
    domain_label=None,
    label_source=LabelSource.PENDING,
    raw_ref="synthetic",
  )


def test_only_statutes_are_loaded(tmp_path: Path) -> None:
  storage = CorpusStorage(tmp_path)
  storage.save_texts(
    [
      document("b", TextSource.STATUTE, "segundo"),
      document("a", TextSource.STATUTE, "primeiro"),
      document("c", TextSource.DJEN, "decisao sintetica"),
      document("d", TextSource.PORTAL, "acordao sintetico"),
    ]
  )
  assert load_articles(tmp_path) == ("primeiro", "segundo")
  assert load_statute_articles(tmp_path) == (("a", "primeiro"), ("b", "segundo"))


def test_case_texts_are_anonymized_djen_once_without_flagged_data(tmp_path: Path) -> None:
  CorpusStorage(tmp_path).save_texts(
    [
      document("a", TextSource.DJEN, "decisão sintética"),
      document("b", TextSource.DJEN, "decisão sintética"),
      document("c", TextSource.DJEN, "contato fulano@exemplo.com.br"),
      document("d", TextSource.STATUTE, "Art. 1º Lei."),
      document("e", TextSource.DJEN, "outra decisão"),
    ]
  )
  assert load_case_texts(tmp_path) == ("decisão sintética", "outra decisão")


def test_split_holds_out_whole_articles() -> None:
  articles = [f"artigo {index}" for index in range(11)]
  split = split_articles(articles)
  assert split.holdout == ("artigo 0", "artigo 5", "artigo 10")
  assert sorted(split.train + split.holdout) == sorted(articles)
  assert not set(split.train) & set(split.holdout)


def test_holdout_characters_never_enter_vocabulary() -> None:
  split = split_articles(["zzz", "a", "b", "c", "d"])
  assert "z" not in build_vocab(split.train).chars


def _imports_lab(path: Path) -> bool:
  for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
    if isinstance(node, ast.Import):
      names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
      names = [f"{node.module}.{alias.name}" for alias in node.names]
    else:
      continue
    for name in names:
      root, _, rest = name.partition(".")
      if root == "agavus" and rest.partition(".")[0] in LAB_PACKAGES:
        return True
  return False


def test_product_never_imports_the_lab_or_its_server() -> None:
  product = [
    path for path in PACKAGE.rglob("*.py") if path.relative_to(PACKAGE).parts[0] not in LAB_PACKAGES
  ]
  product += list((PACKAGE.parents[1] / "space").rglob("*.py"))
  assert product
  assert [path for path in product if _imports_lab(path)] == []
