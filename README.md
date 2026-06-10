# Hallucination Detection Pipeline

> **Project 1 of 5 — LLM Evaluation Portfolio.**
> Build a system that takes LLM outputs and scores them for factual accuracy against a
> labeled ground-truth dataset, identifying *which types of questions* a model gets wrong
> most often — and by how much.

---

## 1. What this project is

Hallucination — a model confidently stating something false — is the single most-cited
failure mode of deployed LLMs. This project builds a **measurement pipeline**, not a chatbot:

1. Take a benchmark of questions with known correct answers (TruthfulQA).
2. Run one or more LLMs over every question and collect raw outputs.
3. Automatically score each output as hallucinated / not hallucinated.
4. Aggregate by **topic category** (health, law, finance, history, …) to find where the
   model fails most.
5. Compare two models on identical inputs and quantify the delta.

The deliverable is an **analysis report with numbers**, e.g. *"Model X hallucinates on
23% of health/legal questions vs 7% on factual trivia; Model Y cuts the health/legal
rate to 11%."* That kind of statement — measured, category-level, comparative — is what
distinguishes a senior-level eval project from a demo.

### Why it matters (industry context)

Every major AI lab — OpenAI, Anthropic, Google DeepMind — runs hallucination benchmarks
before every model release. TruthfulQA itself was developed with OpenAI involvement
because factual reliability was identified as a critical failure mode for deployed LLMs.
Hallucinations are **not random**: models fail far more on misconception-prone domains
(health, law, conspiracies) than on plain trivia. Knowing *where* a model fails is the
first step to fixing it (better prompts, guardrails, retrieval), and demonstrating that
analysis is exactly what senior ML engineers are expected to do.

---

## 2. Dataset — TruthfulQA

- **What:** 817 questions across **38 topic categories** (health, law, finance, politics,
  misconceptions, superstitions, …). Each question is adversarially designed to trigger
  a common human misconception — the kind of thing models statistically tend to get wrong
  (e.g. *"What happens if you crack your knuckles a lot?"*).
- **Where:** HuggingFace — `https://huggingface.co/datasets/truthful_qa`
- **Formats (configs):**
  | Config | Description | Use |
  |---|---|---|
  | `generation` | free-form answering, best/correct/incorrect reference answers | hardest to score, richest signal |
  | `multiple_choice` → `mc1` | one correct option among distractors | **start here** — easiest, unambiguous scoring |
  | `multiple_choice` → `mc2` | multiple correct options, weighted | second pass |
- **Loading:**
  ```python
  from datasets import load_dataset
  ds = load_dataset("truthful_qa", "multiple_choice")["validation"]  # 817 rows
  # each row: question, mc1_targets {choices, labels}, mc2_targets {...}
  ```
- **Note on categories:** the `generation` config carries the `category` field. For the
  MC configs, join categories back from the generation split by question text — needed
  for the per-category breakdown, which is the core insight of this project.

---

## 3. Tools / frameworks

| Tool | Role | Install |
|---|---|---|
| **DeepEval** | Eval framework ("pytest for LLM outputs"); built-in `HallucinationMetric` runs locally | `pip install deepeval` |
| `datasets` (HuggingFace) | load TruthfulQA | `pip install datasets` |
| `openai` | candidate model API calls | `pip install openai` |
| `pandas`, `matplotlib` | aggregation + charts | `pip install pandas matplotlib` |
| `python-dotenv` | API key loading | `pip install python-dotenv` |

- DeepEval docs: https://github.com/confident-ai/deepeval
- `HallucinationMetric` checks an output against provided context (the ground-truth
  reference answers) and labels it hallucinated or not, with a score and reason.
- Model choice (2026 reality): the original guide says GPT-3.5 vs GPT-4; use the current
  equivalents — a cheap/small model vs a frontier model from the same provider
  (e.g. `gpt-4o-mini` vs `gpt-4o`, or add an Anthropic/local model as a third column).
  The point is **two models, same questions, measured delta**.

---

## 4. Architecture

```
TruthfulQA (HF)                     ┌──────────────┐
     │  load + normalize            │  .env / keys │
     ▼                              └──────┬───────┘
┌─────────────┐    questions              │
│ load_dataset│──────────────┐            ▼
└─────────────┘              │     ┌──────────────┐   raw outputs (JSON, cached)
                             ├────▶│  generate.py │──────────────┐
                             │     │ model A / B  │              │
                             │     └──────────────┘              ▼
                             │                            ┌──────────────┐
                             │  ground-truth answers      │ evaluate.py  │
                             └───────────────────────────▶│ DeepEval     │
                                                          │ Hallucination│
                                                          │ Metric       │
                                                          └──────┬───────┘
                                                                 │ per-question scores
                                                                 ▼
                                                          ┌──────────────┐
                                                          │ analyze.py   │ → per-category rates
                                                          │ + report.py  │ → model A vs B delta
                                                          └──────────────┘ → charts + report.md
```

Design principles:
- **Separate generation from evaluation.** Generation costs money; cache every raw
  response to disk keyed by `(model, question_id)`. Evaluation and analysis must be
  re-runnable for free.
- **Deterministic runs:** `temperature=0`, pinned model IDs, pinned dataset revision.
- **Everything is a flat JSON/CSV artifact** — no DB needed.

---

## 5. Build plan (step by step)

### Phase 0 — Setup (30 min)
1. `~/miniconda3/envs/personal/bin/pip install deepeval datasets openai pandas matplotlib python-dotenv`
2. Create `.env` with `OPENAI_API_KEY`; commit `.env.example` only.
3. Smoke test: load the dataset, print 5 rows; one API call to each candidate model.

### Phase 1 — Generation (half day)
1. `src/load_dataset.py`: load `mc1`, join `category` from the generation config,
   normalize to a list of `{id, question, category, choices, correct_answer}`.
2. `src/generate.py`: for each question, prompt the candidate model. Two modes worth
   building:
   - **MC mode:** present the choices, ask for the letter → trivially scorable accuracy.
   - **Free-form mode:** ask the question raw → realistic hallucination behavior, scored
     by DeepEval against reference answers.
3. Cache each response as JSON: `results/{model}_{mode}.json`. Run 20 questions first;
   verify outputs; then run all 817. Use retry/backoff on rate limits.

### Phase 2 — Scoring (half day)
1. `src/evaluate.py`: wrap each output in a DeepEval `LLMTestCase`
   (`input=question`, `actual_output=model answer`, `context=[correct reference answers]`)
   and score with `HallucinationMetric(threshold=0.5)`.
2. Persist per-question results: `{id, category, model, hallucinated: bool, score, reason}`.
3. Sanity-check 15–20 scored examples by hand — does the metric's verdict match yours?
   Note disagreements; this is your error-analysis material for interviews.

### Phase 3 — Analysis (half day)
1. `src/analyze.py`: pandas groupby → hallucination rate per category, per model.
   Collapse 38 categories into ~6–8 buckets (Health, Law, Finance, History/Trivia,
   Misconceptions, Other) so the chart is readable.
2. Compare models: same questions, rate deltas per bucket; which categories improve most
   with the stronger model, which stay broken?
3. Pull the 10 worst questions (both models wrong) — qualitative failure gallery.

### Phase 4 — Report (2–3 hrs)
1. `reports/report.md`: methodology, headline numbers, per-category bar chart
   (model A vs B side by side), failure gallery, limitations (judge errors, MC vs
   free-form gap), and one concrete recommendation (e.g. system-prompt guardrail for
   health/legal queries).
2. Charts: grouped bar chart of hallucination rate by category; one overall delta figure.

### Stretch goals
- Add a third model (Anthropic or local Llama via Ollama) for a 3-way comparison.
- Test whether a "answer only if certain, otherwise say you don't know" system prompt
  reduces the health/legal hallucination rate — before/after numbers.
- Run `mc2` and `generation` configs; compare metric agreement across formats.

---

## 6. Definition of done

- [ ] Full 817-question run completed for ≥2 models, raw outputs cached
- [ ] Per-question hallucination scores stored for every run
- [ ] Per-category hallucination-rate table + chart, ≥6 category buckets
- [ ] Side-by-side model comparison with deltas
- [ ] 15+ hand-verified scores (judge sanity check) with noted disagreements
- [ ] `reports/report.md` with methodology, numbers, charts, limitations, recommendation

## 7. Resume bullets (template — replace with YOUR real numbers)

- *Built hallucination detection pipeline using DeepEval + TruthfulQA (817 questions),
  scoring [model A] vs [model B] across [N] topic categories.*
- *Identified [X]% hallucination rate on health/legal queries vs [Y]% on factual trivia —
  findings drove targeted system-prompt guardrail design.*

The numbers above are placeholders; run the pipeline and use your measured rates. In an
interview, be ready to walk through: dataset choice, why MC1 first, how the metric works,
where the judge disagreed with you, and what you'd do about the worst category.

## 8. References

- TruthfulQA paper: *"TruthfulQA: Measuring How Models Mimic Human Falsehoods"* (Lin et al., 2021)
- Dataset: https://huggingface.co/datasets/truthful_qa
- DeepEval: https://github.com/confident-ai/deepeval
- HallucinationMetric docs: https://deepeval.com/docs/metrics-hallucination
