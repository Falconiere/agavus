---
title: agavus
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: mit
short_description: Classificador jurídico CPU-only com respostas citadas
---

# ⚖️ agavus — demo pública

Classificador + motor de respostas citadas para direito brasileiro (TJDFT),
escopado em **direito de família** e **Maria da Penha / violência doméstica**.

O Space usa o encoder multilingual-e5-small em CPU para encaminhar perguntas e ordenar
artigos de lei. A decisão de aceitar, pedir esclarecimento ou rejeitar considera o custo
de errar a área. Se o encoder estiver indisponível, a resposta é adiada.

Uma resposta aceita traz um trecho citado ou uma conclusão de regra com o artigo que a
fundamenta. Confira a referência, pois a busca pode selecionar um artigo inadequado.
A demonstração usa leis públicas e exemplos sintéticos. A classificação estima a área
da pergunta. Previsão de revogação de medida protetiva está fora do produto.

[Código](https://github.com/Falconiere/agavus) ·
[Classificador de documentos](https://huggingface.co/Falconiere/agavus-classifier)

**Isto não é aconselhamento jurídico; consulte um advogado.**
