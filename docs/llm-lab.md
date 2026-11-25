# LLM lab — learn by building a tiny legal language model

Status: **approved 2026-11-25; no code yet.** An isolated learning track that
builds a tiny transformer with two outputs: next token (LM) and area
(classifier). Companion: [tiny-llm.md](tiny-llm.md), which still governs any
LLM that reaches users.

## 1. Goals and non-goals

Goals, in order:

1. Learn the stack by building it: tokens, loss, sampling, backprop,
   attention, multitask training, evaluation, GPU training.
2. Test one hypothesis: **a shared transformer body with a class head routes
   queries better than TF-IDF + logreg, without losing M recall.**

Non-goals:

- A useful legal assistant trained from scratch (too little data).
- Storing law in weights. Law comes from retrieval; answers stay cited
  excerpts.
- Replacing the live router or changing `answer/`, `decide/` or the Space.

## 2. Isolation rules

| Rule | Why |
| --- | --- |
| Lab code lives in `src/agavus/lab/`; no product package imports it | nothing experimental reaches a user |
| The lab may read product code (evals, corpus readers); never the reverse | one-way dependency, checked by a test |
| PyTorch goes in an optional `lab` extra; steps 1–2 use NumPy or stdlib | the product install stays small |
| Every command runs on CPU; GPU only makes it faster (`--device`) | keeps "no GPU-required code" true |
| Artifacts go to `data/lab/` (gitignored) | same policy as the corpus |
| No lab model is published, even privately, without a design change | weights can memorize training text |

TF-IDF + logreg remains the router. M ceiling preserved: the loss matrix and
decision rule are untouched.

## 3. Data policy

| Data | Lab use | GPU rental |
| --- | --- | --- |
| Statute articles in the corpus (357; ~148 KB) | yes | yes |
| Authored seed queries | yes (class head) | yes |
| Openly licensed pt-BR text | later; license recorded first | yes, if license allows |
| DJEN decisions, DataJud records, anything derived from them | **no** | **never** |
| Gold sets | evaluation only, never training | not needed there |

Why DJEN is excluded: name removal is pattern-based and incomplete, a
5,000-text sample had 9% flagged by the residual scan (some likely false
positives), and language models memorize. Lifting the exclusion needs a
privacy audit, deduplication, a memorization test and a change to this note.

Allowlist, not denylist: the loader selects `source = statute` explicitly.

## 4. Build order

Each step must beat the one before on **held-out loss** (mean `-ln P` of the
true next token), and the result is recorded even when negative. Compare only
runs with the same tokenizer and split; after BPE, report bits per character.

| # | Build | Runs on | Done when |
| --- | --- | --- | --- |
| 0 | Larger gold set: case families, dev/hidden split | – | ready before step 5 |
| 1 | Character bigram, NumPy | CPU | beats uniform, `ln(vocab)` |
| 2 | BPE tokenizer | CPU | encode → decode is lossless |
| 3 | Small neural LM, PyTorch | CPU | beats the bigram |
| 4 | Tiny GPT, ~1M parameters | CPU, then GPU | beats step 3; GPU repeats the CPU result |
| 5 | Same body + area head | CPU or GPU | compared with TF-IDF (§5) |

Stop and review after step 5. Pretrained encoders, RAG and LoRA are separate
decisions, governed by tiny-llm.md.

### Step 1, precisely

- Input: statute articles. Split **by article**, every 5th article held out
  after sorting by id; vocabulary and counts come from the training side only,
  with one `<unk>` character and one boundary token.
- Model: `P(next | current) = (count + k) / (total + k·V)`, with `k = 1`.
- Output: train and held-out loss, uniform baseline, seeded text sample.
- Tests: rows sum to 1; a hand-worked example matches; same seed gives the
  same sample; no held-out article is counted; non-statute sources are refused.

## 5. Step 5: the shared model

```
tokens → embeddings → N blocks ─┬→ LM head     loss_lm
                                └→ class head  loss_cls
loss = loss_lm + λ · loss_cls       (loss_cls only on labeled examples)
```

Areas come from the registry, never literals. Ablations: λ = 0, class head
only, both, and both with the body pretrained on LM first.

Known limit: there are only 13 authored O seeds and no O statutes. Add O
seeds (distinct from gold) before this step, or the result means little.

| Compared with TF-IDF on the same hidden set | Required to claim a win |
| --- | --- |
| M recall | not lower |
| M accepted as F, or M rejected | 0 |
| Macro-F1 | higher, across ≥ 3 seeds |
| Coverage at equal precision | not lower |
| ECE | reported |

A win permits a proposal for shadow mode, not deployment. With few gold items,
report counts and intervals, not just percentages.

## 6. GPU rental (gpubrasil.com.br)

Page checked 2026-11-25; confirm before paying. Prices are per hour:

| GPU | VRAM | R$/h | Use |
| --- | --- | --- | --- |
| RTX 3070 | 8 GB | 0.99 | enough for the ~1M-parameter GPT |
| RTX 3090 | 24 GB | 2.43 | default if the 3070 is unavailable |
| RTX 4090 | 24 GB | 3.60 (also shown as 4.95) | only if measured time justifies it |

The page states SSH + root access, Pix or card, and a first deposit from
R$ 100. It **doesn't state** billing below one hour, disk persistence, or any
LGPD terms. Listed regions are North America, Europe and Asia-Pacific, so
**treat rentals as outside Brazil** (one more reason for §3).

Rules:

1. No rental before step 4 runs on CPU and the GPU script has passed there.
2. First budget: **R$ 100 deposit, at most 5 GPU-hours**. More needs a note
   explaining what was learned. The user pays and creates the machine.
3. Upload only lab code and allowlisted data. No repo tokens, `.env`,
   `corpus.db` or `data/raw/` on the machine; use a dedicated SSH key.
4. Export allowlisted text to a file locally, inspect it, then upload that.
5. Assume the disk can vanish: checkpoint, download, verify checksums.
6. Destroy the machine after each session and confirm billing stopped.
7. Record GPU, hours, cost, seed, config, data hash and losses per run.

## 7. Rule changes (approved 2026-11-25)

- [x] `AGENTS.md`: state that the lab may *use* a GPU but never require one.
- [x] `docs/commits.md` / commit hook: add the `lab` scope.
- [x] `docs/tiny-llm.md`: note this lab as an exception to "never invest in
      training in advance", for learning only.
- [ ] `pyproject.toml`: add the `lab` extra, at step 3 and not earlier.

## 8. Risks

| Risk | Response |
| --- | --- |
| 148 KB of text overfits quickly | expected; early stopping on held-out loss, keep models small |
| Fluent output mistaken for legal truth | lab samples never reach `answer/` |
| The gold set becomes training feedback | dev and hidden splits; hidden set opened per milestone |
| Costs creep | §6 budget and shutdown rules |
| Learning crowds out product fixes | evals (step 0) serve both; retrieval fixes stay a separate track |
