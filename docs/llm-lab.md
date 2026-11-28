# LLM lab — learn by building a tiny legal language model

Status: **approved 2026-11-25; steps 1–4 done (CPU, GPU repeat 2026-11-26). Goal changed
2026-11-27: Agathe becomes a small chat model built from scratch (§9).**
Agathe is the lab’s first named model, a small causal transformer trained to predict the next token.
The lab is isolated from user answers.

## 1. Goals and non-goals

Goals, in order:

1. Learn the stack by building it: tokens, loss, sampling, backprop,
   attention, multitask training, evaluation, GPU training.
2. **Build Agathe into a small chat model from scratch, following the recipe of open chat
   models such as Qwen (§9): our own tokenizer, architecture, pretraining and chat tuning.**
   No pretrained weights. Owner decision 2026-11-27.
3. Test one hypothesis: **a shared transformer body with a class head routes
   queries better than TF-IDF + logreg, without losing M recall.**

Non-goals:

- Matching Qwen's quality: that takes trillions of tokens. Expect GPT-2-level fluency with
  frequent factual errors.
- Replacing the live router or changing `answer/`, `decide/` or the Space. Agathe's answers
  are lab output, never user answers, until a separate design change after evaluation.

## 2. Isolation rules

For scripts written before the package rename, replace `agavus.lab` or
`agavus.language_model` with `agavus.model` in imports and `python -m` commands. The installation
extra is still `lab`, and existing run files are read from `data/lab/`.
In older training commands, replace the `gpt` subcommand with `agathe`.

| Rule | Why |
| --- | --- |
| Training and inference live in `src/agavus/model/`, with a local API in `src/agavus/server/`; no product package imports either | nothing experimental reaches a user |
| The lab may read product code (evals, corpus readers); never the reverse | one-way dependency, checked by a test |
| PyTorch goes in an optional `lab` extra; steps 1–2 use NumPy or stdlib | the product install stays small |
| Every command runs on CPU; GPU only makes it faster (`--device`) | keeps "no GPU-required code" true |
| Artifacts go to `data/lab/` (gitignored) | same policy as the corpus |
| No lab model is published, even privately, without a design change | weights can memorize training text |

TF-IDF + logreg remains the router. M ceiling preserved: the loss matrix and
decision rule are untouched.

### Local Agathe chat

```sh
uv run --extra lab python -m agavus.server
# To select a different checkpoint
uv run --extra lab python -m agavus.server --run data/lab/runs/five-layer-civil-seed0
```

Open `http://127.0.0.1:7861`. One Python process serves the HTML, CSS and JavaScript
in `chat/` and the model API. The lab command `uv run --extra dev python -m agavus.model chat`
starts the same server. Both commands accept `--run DIRECTORY` and `--port PORT`.

The default checkpoint is `data/lab/runs/long-civil`. The server restores `model.pt` on
CPU and checks the exported bundle (`data/lab/export/texts.json`) against the SHA-256
in `run.json` before serving. The default paths follow `AGAVUS_DATA_DIR`.

`GET /api/model` returns the checkpoint name, parameter count and context capacity.
`POST /api/continue` takes a JSON body such as `{"prefix":"Art. "}` and streams one
JSON object per line. Each `text` object contains new characters. The final `metrics`
object reports generation seconds, character count, initial context and capacity.

Enter sends a message. Shift+Enter inserts a newline. The browser stores conversations
and their last response metrics at this address. Clearing site data removes that history.
Agathe generates up to 200 new characters and can end a reply immediately.
The full transcript stays visible. For the two Civil Code checkpoints above,
each prediction uses at most 64 recent characters. The context metric counts the
characters from the conversation supplied at the start of the reply.

## 3. Data policy

| Data | Lab use | GPU rental |
| --- | --- | --- |
| Statute articles in the corpus (347; ~150 KB) | yes | yes |
| Authored seed queries | yes (class head) | yes |
| Openly licensed pt-BR text | training only; license recorded below first | yes, if license allows |
| DJEN decisions as stored in `corpus.db` (anonymized at ingestion), minus texts the residual scan flags | training only | yes, exported from `corpus.db` |
| Raw captures under `data/raw/` (not anonymized) | **no** | **never** |
| Gold sets | evaluation only, never training | not needed there |

**Decision 2026-11-27 (project owner):** the DJEN exclusion is lifted for lab training,
locally and on rented GPUs. Name removal is pattern-based and incomplete (a 5,000-text
sample had 9% flagged by the residual scan, some likely false positives), and language
models memorize, so three safeguards stay: train only on the anonymized `corpus.db` text,
drop every text `contains_personal_data` flags, and keep §2's ban on publishing lab models.
Run a memorization test before any lab model leaves the local machine or the training disk.

Allowlist, not denylist: loaders select `source = statute` or `source = djen` explicitly.

Extra training text is kept under the gitignored data directory.

| Source | License | Size used |
| --- | --- | --- |
| Civil Code articles outside the corpus's family book, from the page already fetched | public domain: laws aren't protected (Lei 9.610/1998, art. 8º, IV) | 1,817 articles, ~524 KB |
| Portuguese Wikipedia, `wikimedia/wikipedia` 20231101.pt via the HF rows API: first 1,500 characters of 350 articles, URL kept per article | CC BY-SA 4.0; attribution kept; no model is published (§2), so nothing is redistributed | ~485 KB |
| 14 federal laws from Planalto, including the Constitution and its transitional provisions. The catalog in `model/federal.py` records each URL. | Public domain under Lei 9.610/1998, art. 8º, IV | 5,257 accepted articles, 3,468,207 UTF-8 bytes, 3,328,868 characters |

| Every federal law Planalto indexes: ordinary, complementary and delegated laws, decree-laws, codes and the Constitution, each in its newest consolidated text (`model/planalto.py`) | Public domain under Lei 9.610/1998, art. 8º, IV | 18,707 law pages found on 2026-11-27; first crawl in progress |
| FineWeb-2 Portuguese (`HuggingFaceFW/fineweb-2`, `por_Latn`): web text filtered by Hugging Face from Common Crawl, pt-BR and pt-PT mixed. Documents the residual personal-data scan flags are dropped | ODC-By 1.0, plus Common Crawl terms of use; attribution kept in the data manifest; no model is published (§2) | first file `000_00000.parquet`, 4.8 GB, of 66 (276 GB) |
| Portuguese Wikipedia 20231101.pt, all six Parquet shards from `wikimedia/wikipedia` | CC BY-SA 4.0; local only, not yet wired into training | 1.58 GB compressed, SHA-256 verified against Hugging Face |

`uv run python -m agavus.model fetch-planalto` walks Planalto's index pages, follows each law
page to its `compilado`/`cons` version when one exists, and keeps the raw HTML with the ETag
and Last-Modified Planalto sent. A rerun sends conditional requests, so it downloads only laws
that changed. Planalto refuses default HTTP clients; the crawler identifies itself as
`Mozilla/5.0 (compatible; agavus-lab/0.1)` and waits half a second between requests. The new
legislation portal lists decree-laws but accepts only browsers: save its two decree-law listing
pages under `data/lab/text/planalto/indexes/` with a browser before crawling.

Extra text only joins the training side. Validation, the exam and the letter
vocabulary stay the statute articles', and any extra text sharing 100
characters in a row with a statute article is dropped, so nothing leaks into
the exam. Wikipedia mixes pt-BR and pt-PT.

`uv run python -m agavus.model fetch-federal` uses the frozen statute reference bundle at
`data/lab/export/texts.json`. Keep this bundle when moving the lab to another Mac.
The collector requests one allowlisted page at a time, with a two-second delay, and reuses
HTML captures under `data/lab/text/federal/sources/`. If the portal times out, rerun the
command. For the first snapshot, direct HTTPS timed out and Firecrawl supplied the official
pages as UTF-8 raw HTML. Keeping the original markup lets the parser remove struck-out text.

Each law gets a JSONL file with its license and the Planalto URL and article number per row.
The collector reports overlap by law and counts unknown characters among the accepted texts.
This snapshot dropped 29 articles and mapped 788 characters to `<unk>` (0.0237%).
Use `agathe federal` or `agathe civil federal` to add the selected sources to training.
An explicit `--texts` selects a self-contained bundle with its own named extras.

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
| 4 | Agathe, 822,627 parameters | CPU, then GPU | beats step 3; GPU repeats the CPU result |
| 4b | Five-layer Agathe, 1,019,875 parameters | CPU or local MPS | lower mean exam loss than step 4 across the same three seeds |
| 5 | Same body + area head | CPU or GPU | compared with TF-IDF (§5) |

Stop and review after step 5. Pretrained encoders, RAG and LoRA are separate
decisions requiring separate evaluation before any production use.

### Step 1, precisely

- Input: statute articles. Split **by article**, every 5th article held out
  after sorting by id; vocabulary and counts come from the training side only,
  with one `<unk>` character and one boundary token.
- Model: `P(next | current) = (count + k) / (total + k·V)`, with `k = 1`.
- Output: train and held-out loss, uniform baseline, seeded text sample.
- Tests: rows sum to 1; a hand-worked example matches; same seed gives the
  same sample; no held-out article is counted; non-statute sources are refused.

### Step 4b comparison

Use identical study text, tokenizer, article splits, optimizer, update budget and checkpoint
selection for both depths. Choose the step on validation for each seed and compare the mean
held-out exam loss. The existing three-seed Civil Code comparison is recorded in §6.
A change in study text requires a new comparison before claiming a win on that mix.

With layer count `L`, width `w`, vocabulary `V` and context `C`, the parameter count is
`L × (12w² + 5w) + 2Vw + Cw + V`. At width 128 with 99 tokens and context 64, an extra
block adds 197,248 parameters. Both depths default to four heads and batch 32.
The head count must divide the width.

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

## 6. Local training and GPU rental

Use the local Mac with MPS first. CPU is the CLI default. The PyTorch wheel on macOS
comes from PyPI. The CPU-only wheel index applies to other platforms. Case-derived text follows
§3 on any hardware: anonymized `corpus.db` text only, never `data/raw/`.

```sh
uv run --extra lab python -m agavus.model agathe civil federal --device mps \
  --layers 5 --width 128 --heads 4 --context 64 --batch 32 --steps 1000 \
  --out data/lab/runs/federal-local
```

Seeded initialization, window selection and sampling use CPU generators. Model tensors
run on the selected device. If a PyTorch version lacks an MPS operation, prefix the command
with `PYTORCH_ENABLE_MPS_FALLBACK=1` to allow that operation to use the CPU.

### Short local benchmark

Apple M2 Max, 96 GiB unified memory, PyTorch 2.15.0. All five measurements used seed 0,
the same 99-token vocabulary and the frozen statute plus Civil Code inputs from bundle
`0cada48c…`. Each run trained for 1,000 updates on the inner training split with the
filtered Civil Code extra. Width was 128, context 64, heads 4, dropout 0.2 and CPU threads 4.
The timer covered model initialization and training, with synchronization at both boundaries.
Validation was scored once afterward. The `agathe` command also selects a checkpoint,
retrains and scores the exam, which adds wall time.

| Device | Layers | Parameters | Batch | Training minutes | Validation loss at 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 4 | 822,627 | 32 | 0.579 | 1.8292 |
| MPS | 4 | 822,627 | 32 | 0.227 | 1.8287 |
| CPU | 5 | 1,019,875 | 32 | 0.729 | 1.7907 |
| MPS | 5 | 1,019,875 | 32 | 0.273 | 1.7792 |
| MPS | 5 | 1,019,875 | 128 | 0.527 | 1.3189 |

At 64 targets per window, batches of 32 and 128 present 2,048 and 8,192 targets per update.
For 20 million presentations on the five-layer MPS model, the measured rates imply 9,766
updates in 0.044 hours or 2,442 updates in 0.021 hours, respectively, plus evaluation and
checkpoint overhead. Random windows can repeat. Batch 128 saw four times as many targets
during this benchmark. These single-seed measurements serve as timing estimates.

### GPU Brasil

Consider renting only when local capacity or measured training time is insufficient.

Operations, disks and safety: **[gpu-brasil.md](gpu-brasil.md)**.

Page checked 2026-11-25; confirm before paying. Prices are per hour:

| GPU | VRAM | R$/h | Use |
| --- | --- | --- | --- |
| RTX 3070 | 8 GB | 0.99 | enough for the five-layer Agathe model |
| RTX 3090 | 24 GB | 2.43 | default if the 3070 is unavailable |
| RTX 4090 | 24 GB | 3.60 (also shown as 4.95) | only if measured time justifies it |

The page states SSH + root access, Pix or card, and a first deposit from
R$ 100. It **doesn't state** billing below one hour, disk persistence, or any
LGPD terms. Listed regions are North America, Europe and Asia-Pacific, so
**treat rentals as outside Brazil** (one more reason for §3).

Rules:

1. No rental before step 4 runs on CPU and the GPU script has passed there.
2. First budget: **R$ 100 deposit, at most 5 GPU-hours**. More needs a note
   explaining what was learned. The user pays; agents act only with approval
   for each action ([gpu-brasil.md](gpu-brasil.md)).
3. Upload only lab code and allowlisted data. No repo tokens, `.env`,
   `corpus.db` or `data/raw/` on the machine; use a dedicated SSH key.
4. Export allowlisted text to a file locally, inspect it, then upload that.
5. Save every run to the `agavus-training` disk (`/workspace/disco/agavus/`).
   It is synced storage, not a backup: also download important checkpoints and
   verify checksums.
6. Destroy the machine after each session and confirm billing stopped.
7. Record GPU, hours, cost, seed, config, data hash and losses per run.
   `agathe` and `sft` print a progress line every 100 steps, at every loss check and at the
   last step (`step 1200/12000  batch 0.4120  0.180 s/step  eta 0h32m`, plus `train` and
   `held-out` at checks), flushed, so `tail -f` on the run log shows where a run stands.

### Run log

**Session 1, 2026-11-26.** RTX 3070 8 GB, PyTorch 2.15.0+cu132, ~55 min,
R$ 1.03. Default `AgatheConfig` (822,627 parameters, context 64, width 128,
4 layers, 4 heads, dropout 0.2). Data: exported bundle, sha256 `0cada48c…`.
Runs on `agavus-training` under `runs/`, copies in `data/lab/runs/`
(checksums verified). The GPU repeats the CPU result within seed noise.

| Run | Seed | Steps tried / chosen | Best validation | Exam | Minutes |
| --- | --- | --- | --- | --- | --- |
| statutes | 0, 1, 2 | 6,000 / 5,500, 5,500, 4,000 | 1.141, 1.122, 1.143 | 1.162, 1.169, 1.161 | 6–7 |
| + Civil Code | 0, 1, 2 | 6,000 / 5,500, 6,000, 6,000 | 1.169, 1.185, 1.178 | 1.226, 1.171, 1.192 | 6–7 |
| + Civil Code | 0 | 30,000 / 26,000 | 1.057 | **1.093** | 30 |

CPU references, seed 0: statutes 1.181, + Civil Code 1.174 (6,000 steps).
Lessons: one run can be off by ~0.05, so compare seed averages; extra text
only pays off with more steps.

**CPU depth comparison, 2026-11-26.** Agathe accepts `--layers` and defaults
to four blocks. Both depths used the same statute and Civil Code bundle
(`0cada48c…`), article splits, 6,000-step budget and 500-step validation interval.
Each checkpoint was reloaded and scored on the held-out statute articles.

| Layers | Parameters | Seed 0 loss | Seed 1 loss | Seed 2 loss | Mean loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 822,627 | 1.1990 | 1.1884 | 1.1829 | 1.1901 |
| 5 | 1,019,875 | 1.1528 | 1.1662 | 1.1693 | 1.1628 |

The fifth block reduced mean next-character loss by 0.0274 on this split.
Validation selected 5,500 steps for the four-layer seed-0 run and 6,000 for
the other five runs. Weights and run records are under
`data/lab/runs/{four,five}-layer-civil-seed{0,1,2}/`.

### Memory investigation, 2026-11-27

**Training memory.** RTX 4090, 8 layers, width 512, 8 heads, about 16,384 tokens per step,
peak allocated memory under a 45% per-process cap while two runs shared the card:

| Context | Batch | Explicit attention, fp32 | Fused attention, fp32 | Fused attention, bf16 |
| ---: | ---: | --- | --- | --- |
| 256 | 64 | 6.6 GiB, 46k tokens/s | 5.5 GiB, 62k tokens/s | 3.8 GiB, 112k tokens/s |
| 1,024 | 16 | out of memory | 8.5 GiB, 50k tokens/s | 3.8 GiB, 108k tokens/s |
| 4,096 | 4 | out of memory | 4.7 GiB, 28k tokens/s | 3.8 GiB, 87k tokens/s |
| 16,384 | 1 | out of memory | 4.8 GiB, 10k tokens/s | 3.9 GiB, 54k tokens/s |

Agathe now trains with fused attention (`--bf16` adds mixed precision); the explicit table is
built only to show attention. Context length is no longer limited by GPU memory on this card.

**Long-term memory.** `python -m agavus.model recall` prompts with the first 64 characters of a
text and compares the greedy next 64. `long-civil` (822,627 parameters, context 64) reproduced
none of 80 training articles exactly and matched 4.9% of answer characters before the first
mistake (exam: 63 articles, 0%, 6.0%). Training and exam loss were close (0.978 and 1.025), so
the model generalizes letter patterns but stores almost no wording. The same probe on `djen`
is the memorization test §3 requires before a DJEN-trained model leaves this machine.

**Working memory in chat.** The chat sends only the last `context` characters of the whole
conversation, so `long-civil` sees 64 characters. Each new character reruns the model over the
full window; long windows need a key/value cache before chat with a 4k-16k context is usable on
CPU.

## 7. Rule changes (approved 2026-11-25)

- [x] `AGENTS.md`: state that the lab may *use* a GPU but never require one.
- [x] `docs/commits.md` / commit hook: add the `lab` scope.
- The lab may train models from scratch for learning. Production model investments require
  a measured gap after data, feature and calibration improvements.
- [x] `pyproject.toml`: add the `lab` extra, at step 3 and not earlier.

## 8. Risks

| Risk | Response |
| --- | --- |
| 150 KB of text overfits quickly | expected; early stopping on held-out loss, keep models small |
| Fluent output mistaken for legal truth | lab samples never reach `answer/` |
| The gold set becomes training feedback | dev and hidden splits; hidden set opened per milestone |
| Costs creep | §6 budget and shutdown rules |
| Learning crowds out product fixes | evals (step 0) serve both; retrieval fixes stay a separate track |

## 9. Agathe chat: the from-scratch recipe

A model that only predicts the next character continues text; it answers only after chat
tuning on conversations. Each stage gets measured before the next one starts.

| Stage | What to build | Done when |
| --- | --- | --- |
| 9.1 Tokenizer | Byte-level BPE, about 32k tokens, trained with the `tokenizers` library on our corpus; special tokens for turns (`<|user|>`, `<|assistant|>`, `<|end|>`) | encode → decode is lossless on every source; report characters per token |
| 9.2 Architecture | RoPE positions, RMSNorm, SwiGLU, grouped-query attention, tied embeddings; fused attention and bf16 already exist | a short run matches or beats the current Agathe at equal compute, in bits per character |
| 9.3 Pretraining data | Planalto laws, anonymized DJEN text, full Portuguese Wikipedia, and openly licensed Portuguese web text (license recorded in §3 first) | 2–5 billion tokens, deduplicated, exam statutes held out |
| 9.4 Pretraining | about 100–300M parameters, single pass with validation | bits per character on the statute exam; recall probe on DJEN shows no verbatim leaks |
| 9.5 Chat tuning (SFT) | Conversations generated by our rules from the texts, for example "O que diz o art. 22 da Lei 11.340?" → the article with its citation, plus conversations written by a teacher LLM. The teacher sees only public law text, never DJEN text | the model answers in turn format on held-out questions |
| 9.6 Evaluation | Gold questions from `evals/`, citation accuracy, recall and memorization probes | numbers recorded here, including failures |

Chat tuning (9.5, `model/sft.py`): each pair is rendered as
`<|user|>question<|end|><|assistant|>answer<|end|>`, and the loss counts only the answer
tokens and the end token that closes each answer, so the model learns to answer, not to ask.
The rules (`model/sft_rules.py`) read each Planalto law's name and ementa from its capture
and fill four Portuguese paraphrases per template: an article's text with "Fonte: <lei>,
art. N", what a law covers with its ementa, and which law a short ementa describes. Each
conversation keeps its `origin_ref` and the public-domain license. Every tenth law by URL
is held out whole for evaluation, so no law sits on both sides.
`uv run python -m agavus.model build-sft` writes `data/lab/sft/rules-train.jsonl` and
`rules-eval.jsonl`; `uv run python -m agavus.model sft --run <base> --out <dir>` tunes a run
on them (rate 1e-4 by default) and saves it with `"chat": true`. The local chat then sends
the turns and stops at `<|end|>`.

Agathe answers from retrieved articles placed in its context, not from memory
(`model/law_search.py`, `model/sft_context.py`). `build-search` indexes every crawled Planalto
article in SQLite FTS5 (BM25, accents folded) at `data/lab/text/planalto/search.db`, keeping
each row's `origin_ref` and law name; each law's ementa is a row too, cited by the law's url
and labeled "<lei>, ementa". Words in every name or article ("lei", "art", "nº") are not
searched. A question that names a law by kind and number ("art. 49 da Lei 8.078", "Lei
Complementar nº 101") gets that law's asked article first, or its ementa when no article is
asked, then BM25 within that law, then BM25 over all laws. `build-sft-context [--teacher FILE]`
searches each rule (and teacher) question, puts the top three hits, at most 1,200 characters
each, numbered before `Pergunta:` in the user turn, and adds the cited article or ementa when
search misses it, so ementa and "which law" pairs train too. Two seeded shares answer "Os
textos encontrados não respondem a essa pergunta.", so the model learns to defer: one in ten
has the cited text taken out, and one in twenty (`--unrelated`) is shown the hits for another
law's question. Search costs about 0.25 s per rule question on one core; `--workers N` splits
the build into N interleaved shards, shard i seeded `seed + 1000·i` (one worker equals the
serial build). Teacher pairs join the train side only, and pairs from held-out laws are
dropped. It writes `data/lab/sft/context-train.jsonl` and `context-eval.jsonl`; tune with
`sft --run <base> --out <dir> --train <context-train> --eval <context-eval>`.

The chat server (`--search PATH`, this index by default) wraps the last question the same
way when the run is chat-tuned and the index exists, lists the articles as "Fontes consultadas", allows
answers up to 2,000 characters, and stops an answer that repeats one short token cycle six
times. Chat answers decode greedily (the likeliest token, so quotes copy the context), and
with the index nothing streams until the answer passes `model/grounding.py`: at least 80% of
its word 4-grams must appear in the shown articles' full text and names, its "Fonte:" line
must cite a shown article, and a question naming a law or article by number must get that
one cited. Otherwise, or after a loop, the chat answers "Os textos encontrados não respondem
a essa pergunta." and still lists the articles consulted. A plain "which law amends Lei X"
question therefore defers, since its answer carries no "Fonte:" line.

**Quoted answers (default when the reranker is installed).** A 100M model trained from
scratch paraphrases and invents even with the right articles in context, and its own
likelihoods do not tell a relevant passage from an easy-to-copy one (tried 2026-11-28: per-token
loss and question-vs-neutral PMI both picked wrong passages and failed to defer). So the chat
quotes instead of generating (`model/chat_search.py`, `model/passages.py`, `model/reranker.py`):

- Search: the domain statutes (Lei 11.340, 12.318, 13.058, 14.994, Civil Code família book)
  come from the product's `corpus.db` through its E5 statute search, cited by their Planalto url
  and name; the rest from `search.db` by BM25. Training reads neither path: the exam stays clean.
- Passages: each article whole, its caput with incisos, and each paragraph.
- Reranker: pretrained `BAAI/bge-reranker-v2-m3` (Apache-2.0), onnx-community int8 export pinned
  by revision, on CPU (`python -m agavus.model fetch-reranker`, ~560 MB in `data/models/`). It
  reads question and passage (under its citation) together.
- Answer: the best passage verbatim (whitespace collapsed) with "Fonte:" built from its article;
  below a logit of 1.0 the chat defers. A question naming law and article ("art. 22 da Lei Maria
  da Penha") is a lookup: that article is quoted whole and nothing else competes.

Calibration (2026-11-28): on 300 held-out rule questions the reranker's best passage came from the
cited article 99.6% of the time (mMiniLM-mMARCO, the smaller alternative, 96.9%); on 600 natural
teacher questions, 89%, and the 1.0 threshold kept 96% of answerable ones while off-topic
questions score near -11. On the authored retrieval DEV set (40 F+M questions, `evals/retrievalset.py`; HIDDEN untouched)
quoted answers were 25 correct, 4 wrong and 11 deferred with 8 domain articles per question
(3 articles: 22/6/12; a relevant article was among the hits for 34 of 40). Dropping Planalto laws
that only amend a domain statute changed nothing; a 0.5 threshold gave 27/5/8, so 1.0 stays.
Known gap: plain-language questions whose law BM25 never retrieves
(férias do servidor → Lei 8.112) get a confident wrong-law quote; dense search over all of
Planalto is the next step.

`eval-chat --run <tuned run> [--limit N] [--workers N]` measures this offline (CPU): each
held-out context conversation is answered greedily from the same user turn, the shown
articles are read back whole from the training index, and the answer goes through the same
check. Answerable questions end correct (same citation, body copied from the gold answer),
deferred or wrong-shown; NOT_FOUND questions end deferred or answered. Every case goes to
`<run>/eval-chat.jsonl`.

First context run (2026-11-27, `q-chat-context-100m`, 12k steps on q-planalto-djen-100m-60k,
RTX 4090, ~40 min): masked held-out loss 0.281 nats per answer token on 1,621 held-out article
questions. Chat test: art. 49 da Lei 8.078 quoted correctly; "Do que trata a Lei 8.078" and
férias do servidor answered from the wrong law (search ranked Leis 14.078 and 4.078 first and
missed Lei 8.112; ementa pairs were absent); "como revogar uma medida protetiva" looped
("de, de, …") instead of deferring. The changes above target the ementa, wrong-law and looping
failures; plain-language questions such as férias still miss Lei 8.112 in BM25.

Second context run (2026-11-27, `q-chat-context-v2-100m`, same base and steps on the 78,029
conversations above, RTX 4090, ~50 min): masked held-out loss 0.251 on 2,000 held-out
questions. Before the grounding check, art. 49, "Do que trata a Lei 8.078" and "Qual lei
dispõe sobre a proteção do consumidor" were answered correctly, while férias and "como
revogar uma medida protetiva" were invented (4-gram coverage 0.20 and 0.09). With greedy
decoding and the check (on CPU), those two, a cooking question and "art. 22 da Lei 11.340"
(not indexed; the model quoted art. 46 instead) defer, and the three correct answers stay.

`eval-chat` on all 2,000 held-out questions (2026-11-28, 8 CPU workers, ~35 min):

| Kind | Correct | Deferred, answerable | Wrong shown | Deferred negative | Answered negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| article | 1,270 | 109 | 0 | 241 | 1 |
| ementa | 163 | 1 | 0 | 27 | 0 |
| reverse ("qual lei…") | 123 | 39 | 4 | 13 | 9 |

Answerable 1,709: 1,556 correct (91%), 149 deferred, 4 wrong shown; negatives 291: 10
answered. The check blocked 205 written answers (7 loops). Most errors are reverse questions,
whose answers carry no "Fonte:" line, so only coverage guards them; long articles defer when
the answer is cut at 2,000 characters before its "Fonte:" line.

Agathe in `model/agathe.py` now uses this architecture; character-level runs before
2026-11-27 are archived (loadable from git history). Parameters: `V·d + L·(2d + 2d² +
2d·k·h + 3d·f) + d`, with vocabulary V, width d, layers L, key/value heads k, head size
h = d / heads, and SwiGLU width f = the multiple of 64 nearest 8d/3 (two RMSNorm gains,
query and output mix, shared key and value, gate/up/down per layer; the token table is
also the output). Runs record the tokenizer SHA-256 and report validation and exam in
bits per character.

The GPU for 9.4 is chosen when 9.1–9.3 are ready (§6 rules apply). Safety ceiling and the
product's cited-excerpt rule are unchanged: M ceiling preserved.
