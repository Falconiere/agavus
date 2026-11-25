"""agavus Gradio Space — public demo.

Privacy by construction: only public-domain statutes (bundled in assets/)
+ synthetic seeds are bootstrapped here; no real case data ever enters this
Space (LGPD). Startup downloads pinned public encoder weights; without them
the engine falls back to word search. Questions never leave the Space.
"""

from functools import cache
from pathlib import Path

import gradio as gr

from agavus.answer.compose import compose
from agavus.classify.store import load_model
from agavus.config.settings import load_settings
from agavus.corpus.storage import CorpusStorage
from agavus.decide.rule import decide
from agavus.publish.bootstrap import ensure_model
from agavus.retrieve.dense import prepare_encoder
from agavus.retrieve.encoder import Encoder

ASSETS = Path(__file__).parent / "assets"
EXAMPLES = [
  "Qual é o artigo da lei de alienação parental?",
  "ele descumpriu a medida protetiva, o que acontece?",
  "como funciona habeas corpus?",
]


@cache
def _bootstrap(data_dir: Path) -> Path:
  """Import bundled public statutes, label, train — once per data dir."""
  return ensure_model(load_settings(), ASSETS)


@cache
def _encoder(data_dir: Path) -> Encoder | None:
  """Encoder over the bootstrapped statutes — prepared once per data dir."""
  return prepare_encoder(data_dir)


def answer_question(question: str) -> tuple[str, str, str]:
  """Return (decision header, cited answer, audit trace) for the UI."""
  if not question.strip():
    return ("", "Digite uma pergunta.", "")
  settings = load_settings()
  classifier = load_model(_bootstrap(settings.data_dir), "classifier")
  probs = classifier.predict_proba([question])[0]
  mapping = {
    label: float(value) for label, value in zip(classifier.model.classes, probs, strict=True)
  }
  answer = compose(question, mapping, CorpusStorage(settings.data_dir), _encoder(settings.data_dir))
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


def warm_up() -> None:
  """Do the slow work before serving, so no visitor waits for it."""
  data_dir = load_settings().data_dir
  _bootstrap(data_dir)
  _encoder(data_dir)


if __name__ == "__main__":
  warm_up()
  build_demo().launch()
