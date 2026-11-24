"""agavus Gradio Space — public demo.

Privacy by construction: only public-domain statutes (bundled in assets/)
+ synthetic seeds are bootstrapped here; no real case data ever enters this
Space (LGPD). No network needed at startup.
"""

from functools import cache
from pathlib import Path

import gradio as gr

from agavus.answer.compose import compose
from agavus.classify.pipeline import train_classifier, training_set
from agavus.classify.store import load_model, save_model
from agavus.config.settings import load_settings
from agavus.corpus.ingest import import_statute_file
from agavus.corpus.storage import CorpusStorage
from agavus.decide.rule import decide

ASSETS = Path(__file__).parent / "assets"
ASSET_NAMES = {
  "l11340": "Lei 11.340/2006",
  "l12318": "Lei 12.318/2010",
  "l13058": "Lei 13.058/2014",
  "l14994": "Lei 14.994/2024",
  "l10406": "Lei 10.406/2002",
}
EXAMPLES = [
  "Qual é o artigo da lei de alienação parental?",
  "ele descumpriu a medida protetiva, o que acontece?",
  "como funciona habeas corpus?",
]


@cache
def _bootstrap(data_dir: Path) -> None:
  """Import bundled public statutes, label, train — once per data dir."""
  settings = load_settings()
  db_path = data_dir / "models" / "models.db"
  if db_path.exists():
    return
  for stem, name in ASSET_NAMES.items():
    import_statute_file(settings, name, ASSETS / f"{stem}.html")
  storage = CorpusStorage(data_dir)
  storage.run_labeling()
  texts, targets = training_set(storage)
  classifier, report = train_classifier(texts, targets)
  save_model(db_path, "classifier", classifier, report)


def answer_question(question: str) -> tuple[str, str, str]:
  """Return (decision header, cited answer, audit trace) for the UI."""
  if not question.strip():
    return ("", "Digite uma pergunta.", "")
  settings = load_settings()
  _bootstrap(settings.data_dir)
  classifier = load_model(settings.data_dir / "models" / "models.db", "classifier")
  probs = classifier.predict_proba([question])[0]
  mapping = {
    label: float(value) for label, value in zip(classifier.model.classes, probs, strict=True)
  }
  answer = compose(question, mapping, CorpusStorage(settings.data_dir))
  decision = decide(mapping)
  header = f"**Decisão:** `{answer.decision}` · confiança {max(mapping.values()):.2f}"
  body = f"```\n{answer.text}\n```"
  losses = " · ".join(f"{name} {value:.2f}" for name, value in decision.expected_losses.items())
  features = ", ".join(
    f"{term} ({value:+.2f})" for term, value in classifier.top_features(question)
  )
  probabilities = " · ".join(f"{label} {value:.3f}" for label, value in mapping.items())
  trace = (
    f"probabilidades: {probabilities}\n\nperdas esperadas: {losses}\n\ntop features: {features}"
  )
  return (header, body, trace)


def build_demo() -> gr.Blocks:
  """Assemble the Gradio UI (no launch at import time)."""
  demo: gr.Blocks
  with gr.Blocks(title="agavus") as demo:
    gr.Markdown(
      "# ⚖️ agavus\n"
      "Classificador jurídico **CPU-only** com respostas citadas — TJDFT, "
      "direito de família e Maria da Penha. Sem LLM no núcleo: matemática "
      "calibrada + decisões de três vias (aceitar / adiar / rejeitar).\n\n"
      "**Isto não é aconselhamento jurídico; consulte um advogado.**"
    )
    question = gr.Textbox(label="Pergunta", placeholder=EXAMPLES[0])
    button = gr.Button("Perguntar", variant="primary")
    header = gr.Markdown()
    answer = gr.Markdown()
    with gr.Accordion("Explicação (auditoria)", open=False):
      trace = gr.Markdown()
    button.click(answer_question, inputs=question, outputs=[header, answer, trace])
    question.submit(answer_question, inputs=question, outputs=[header, answer, trace])
    gr.Examples(EXAMPLES, inputs=question)
    gr.Markdown(
      "[Código (MIT)](https://github.com/Falconiere/agavus) · "
      "[Modelo no Hub](https://huggingface.co/Falconiere/agavus-classifier)"
    )
  return demo


if __name__ == "__main__":
  build_demo().launch()
