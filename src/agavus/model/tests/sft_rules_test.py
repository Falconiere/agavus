"""Rule conversations from synthetic Planalto-like pages: cards, templates, hold-out by law."""

import json
from pathlib import Path

from agavus.model.federal import PUBLIC_DOMAIN
from agavus.model.planalto import PLANALTO_DIR, SITE
from agavus.model.sft_rules import (
  ARTICLE_QUESTIONS,
  EMENTA_QUESTIONS,
  REVERSE_QUESTIONS,
  LawCard,
  article_number,
  law_card,
  law_of,
  load_rule_sources,
  rule_conversations,
  split_by_law,
)

LAW = SITE + "leis/l99999.htm"
EMENTA = "Dispõe sobre medidas protetivas fictícias e dá outras providências."
PAGE = """<html><head><title>Lei nº 99.999</title></head><body>
<p>Presidência da República</p><p>Casa Civil</p>
<p>LEI Nº 99.999, DE 1º DE JANEIRO DE 2030</p>
<p>Mensagem de veto</p>
<p>(Vide Lei nº
  99.998, de 2030)</p>
<p>Dispõe sobre medidas protetivas
  fictícias e dá outras providências.</p>
<p>(Redação dada pela Lei nº 99.997, de 2031)</p>
<p>O PRESIDENTE DA REPÚBLICA</p>
<p>Faço saber que o Congresso Nacional decreta e eu sanciono a seguinte Lei:</p>
<p>Art. 1º Esta Lei cria medidas fictícias.</p>
</body></html>"""
DECREE = """<p>DECRETO-LEI N<sup>o</sup> 1.234, DE<br>3 DE MARÇO DE<br>1940.</p>
<p>Cria um conselho fictício.</p><p>O Presidente da República, usando das atribuições,</p>
<p>Art. 1º Fica criado o conselho.</p>"""
CITED_ONLY = """<p>Altera normas.</p><p>Art. 1º O art. 2º da</p>
<p>Lei nº 1.111, de 2 de maio de 1990, passa a vigorar assim.</p>"""
TEMPLATES = 4
LAWS = 25
ARTICLES = [(f"{LAW}, art. {n}", f"Art. {n}º Texto fictício número {n}.") for n in range(1, 9)]


def _cards(count: int) -> list[LawCard]:
  return [
    LawCard(f"{SITE}leis/l{n:05d}.htm", f"Lei nº {n}, de 1º de maio de 2030", "Cria x.")
    for n in range(count)
  ]


def test_law_card_reads_the_heading_and_the_ementa() -> None:
  assert law_card(LAW, PAGE.encode()) == LawCard(
    LAW, "Lei nº 99.999, de 1º de janeiro de 2030", EMENTA
  )
  decree = law_card(LAW, DECREE.encode())
  assert decree == LawCard(
    LAW, "Decreto-Lei nº 1.234, de 3 de março de 1940", "Cria um conselho fictício."
  )
  assert law_card(LAW, CITED_ONLY.encode()) is None


def test_article_numbers_follow_the_article_text() -> None:
  assert article_number(f"{LAW}, art. 1", "Art. 1º Esta Lei.") == "1º"
  assert article_number(f"{LAW}, art. 12-C", "Art. 12-C. Fica.") == "12-C"
  assert article_number(f"{LAW}, art. 7", "Sem cabeçalho.") == "7"


def test_rule_conversations_are_seeded_cited_and_varied() -> None:
  card = LawCard(LAW, "Lei nº 99.999, de 1º de janeiro de 2030", EMENTA)
  found = rule_conversations([card], {LAW: ARTICLES}, seed=3)
  assert found == rule_conversations([card], {LAW: ARTICLES}, seed=3)
  assert [c.source for c in found] == ["rules:article"] * 8 + ["rules:ementa", "rules:reverse"]
  assert all(c.license == PUBLIC_DOMAIN for c in found)
  assert [c.origin_ref for c in found] == [ref for ref, _ in ARTICLES] + [LAW, LAW]
  first = found[0].turns
  assert first[1] == ("assistant", f"{ARTICLES[0][1]}\n\nFonte: {card.name}, art. 1º.")
  assert len({c.turns[0][1].split(" art")[0] for c in found[:8]}) > 1
  question, answer = found[-1].turns
  assert answer == ("assistant", f"{card.name}.")
  assert "dispõe sobre medidas protetivas fictícias e dá outras providências?" in question[1]
  assert min(map(len, (ARTICLE_QUESTIONS, EMENTA_QUESTIONS, REVERSE_QUESTIONS))) >= TEMPLATES


def test_decree_laws_read_in_the_masculine_and_long_ementas_skip_the_reverse() -> None:
  card = LawCard(LAW, "Decreto-Lei nº 1.234, de 3 de março de 1940", "Cria " + "x" * 300)
  found = rule_conversations([card], {LAW: ARTICLES[:1]}, seed=0)
  assert [c.source for c in found] == ["rules:article", "rules:ementa"]
  questions = [conversation.turns[0][1] for conversation in found]
  assert " do Decreto-Lei" in questions[0]
  assert not any(" da Decreto" in text or " a Decreto" in text for text in questions)


def test_split_by_law_holds_out_whole_laws() -> None:
  cards = _cards(LAWS)
  articles = {card.url: [(f"{card.url}, art. 1", "Art. 1º X.")] for card in cards}
  train, held = split_by_law(rule_conversations(cards, articles, seed=0))
  train_laws, held_laws = {law_of(c) for c in train}, {law_of(c) for c in held}
  assert not train_laws & held_laws and len(train_laws | held_laws) == LAWS
  assert held_laws == {cards[9].url, cards[19].url}


def test_load_rule_sources_reads_the_capture_the_articles_came_from(tmp_path: Path) -> None:
  folder = tmp_path / PLANALTO_DIR
  compiled = SITE + "leis/l99999compilado.htm"
  for url, page in ((LAW, DECREE), (compiled, PAGE)):
    capture = folder / "sources" / f"{url.removeprefix(SITE)}.html"
    capture.parent.mkdir(parents=True, exist_ok=True)
    capture.write_text(page, encoding="utf-8")
    capture.with_suffix(".json").write_text("{}", encoding="utf-8")
  row = {"license": PUBLIC_DOMAIN, "url": compiled, "origin_ref": f"{compiled}, art. 1"}
  articles = folder / "articles" / "leis" / "l99999.htm.jsonl"
  articles.parent.mkdir(parents=True)
  articles.write_text(json.dumps(row | {"text": "Art. 1º X."}) + "\n", encoding="utf-8")
  cards, by_url = load_rule_sources(tmp_path)
  assert [card.name for card in cards] == ["Lei nº 99.999, de 1º de janeiro de 2030"]
  assert by_url == {compiled: [(f"{compiled}, art. 1", "Art. 1º X.")]}
