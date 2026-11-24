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
short_description: Classificador jurídico CPU-only com respostas citadas (TJDFT)
---

# ⚖️ agavus — demo pública

Classificador + motor de respostas citadas para direito brasileiro (TJDFT),
escopado em **direito de família** e **Maria da Penha / violência doméstica**.

- Núcleo 100% CPU: TF-IDF + regressão logística calibrada + decisão de três
  vias (aceitar / adiar / rejeitar) com matriz de custos sensível à segurança.
- Respostas são **trechos verbatim citados**, nunca opiniões geradas.
- Incerteza é adiada, não chutada. Fora do escopo é dito claramente.
- Privacidade: apenas leis de domínio público + seeds sintéticos neste Space
  (LGPD — nenhum dado processual real).

Código: https://github.com/Falconiere/agavus ·
Modelo: https://huggingface.co/Falconiere/agavus-classifier

**Isto não é aconselhamento jurídico; consulte um advogado.**