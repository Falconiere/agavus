"""Lab data policy: statutes only, whole-article split, one-way imports."""

import ast
from pathlib import Path

import agavus
from agavus.corpus.records import LabelSource, TextDocument, TextSource
from agavus.corpus.storage import CorpusStorage
from agavus.lab.bigram import build_vocab
from agavus.lab.data import load_articles, split_articles

PACKAGE = Path(agavus.__file__).parent


def _document(name: str, source: TextSource, text: str) -> TextDocument:
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
      _document("b", TextSource.STATUTE, "segundo"),
      _document("a", TextSource.STATUTE, "primeiro"),
      _document("c", TextSource.DJEN, "decisao sintetica"),
      _document("d", TextSource.PORTAL, "acordao sintetico"),
    ]
  )
  assert load_articles(tmp_path) == ("primeiro", "segundo")


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
      if any(alias.name.startswith("agavus.lab") for alias in node.names):
        return True
    elif isinstance(node, ast.ImportFrom):
      module = node.module or ""
      if module.startswith("agavus.lab"):
        return True
      if module == "agavus" and any(alias.name == "lab" for alias in node.names):
        return True
  return False


def test_product_never_imports_lab() -> None:
  lab = PACKAGE / "lab"
  product = [path for path in PACKAGE.rglob("*.py") if lab not in path.parents]
  product += list((PACKAGE.parents[1] / "space").rglob("*.py"))
  assert product
  assert [path for path in product if _imports_lab(path)] == []
