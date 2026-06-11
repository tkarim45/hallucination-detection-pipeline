# Hallucination Detection Pipeline

**Measuring where LLMs confidently get it wrong — per topic, per model, with numbers.**

An end-to-end evaluation pipeline that runs two Claude models (via AWS Bedrock) over the
[TruthfulQA](https://huggingface.co/datasets/truthful_qa) benchmark (817 adversarial
questions, 38 topic categories), scores every answer as hallucinated / not hallucinated,
and breaks failure rates down by topic category — so the output isn't "the model
hallucinates sometimes" but *"Model X hallucinates on N% of health/legal questions vs
M% on factual trivia, and Model Y cuts that by Z points."*

> 🚧 **Full 817-question run in progress** — headline numbers, per-category charts, and
> the model-vs-model delta land here when it completes. Pilot run (20 questions) is
> fully wired end-to-end.

## Why this exists

Hallucination is the most-cited failure mode of deployed LLMs, and it is **not random**:
models fail far more on misconception-prone domains (health, law, conspiracies) than on
plain trivia. Knowing *where* a model fails is the first step to fixing it — better
prompts, guardrails, retrieval. Every major lab runs exactly this kind of benchmark
before a model release; this repo is a from-scratch, reproducible version of that
workflow.

## How it works

```
TruthfulQA (HuggingFace)
      │  load mc1 + join categories from generation config
      ▼
┌──────────────┐   questions    ┌─────────────────────────┐
│load_dataset.py│──────────────▶│ generate.py             │  AWS Bedrock
└──────────────┘                │ Claude Haiku 4.5        │  temperature=0
      │                         │ Claude Sonnet 4.6       │  per-response cache
      │ reference answers       └───────────┬─────────────┘
      │                                     │ raw outputs (JSON)
      ▼                                     ▼
┌─────────────────────────────────────────────────────────┐
│ evaluate.py                                             │
│  • MC mode: exact-match letter scoring (no judge)       │
│  • Free-form: DeepEval HallucinationMetric,             │
│    judge = Claude Opus 4.6 (stronger than both          │
│    candidates → less judge bias)                        │
└───────────────────────────┬─────────────────────────────┘
                            │ per-question scores
                            ▼
              ┌──────────────────────────┐
              │ analyze.py + report.py   │ → 38 categories → 7 buckets
              │                          │ → hallucination rate per bucket per model
              └──────────────────────────┘ → charts + reports/report.md
```

**Two evaluation modes, deliberately:**

| Mode | What it measures | Scoring |
|---|---|---|
| `mc` | Can the model pick the true statement among misconceptions? | Exact-match — cheap, unambiguous, no judge error |
| `free` | Does the model hallucinate when answering naturally? | LLM-as-judge (DeepEval) vs reference answers — realistic but judge-dependent |

The gap between the two modes is itself a finding.

## Engineering decisions

- **Generation separated from evaluation.** Generation costs money; every raw response
  is cached to disk keyed by `(model, question_id)` **after each call**, so an
  interrupted run (throttle storm, network, Ctrl-C) resumes with zero re-spend.
  Scoring and analysis are derived artifacts — re-runnable for free, forever.
- **Deterministic runs.** `temperature=0`, pinned Bedrock model IDs, raw outputs frozen
  before scoring.
- **Throttle-resilient.** Fresh AWS accounts get tight Bedrock quotas; the pipeline
  survives sustained 429 storms with exponential backoff up to 2-minute waits instead
  of dying mid-run.
- **Judge ≠ candidate.** The DeepEval judge (Opus 4.6) is a strictly stronger model
  than both candidates, and MC mode provides a judge-free baseline to sanity-check
  judge verdicts against.

## Cost

Full 817-question run, both models, both modes, measured from actual token usage:

| Item | Cost |
|---|---|
| Generation (4 × 817 questions) | ~$1.64 |
| LLM-judge scoring (2 × 817 free-form answers, Opus 4.6) | ~$17 |
| **Total** | **~$19** |

Throttled/retried requests are never billed; wall-clock time is quota-bound, not
cost-bound.

## Quickstart

**Prereqs:** Python 3.12, an AWS account with Bedrock model access enabled for
Anthropic Claude models (Console → Bedrock → Model access), and an IAM user with
`AmazonBedrockFullAccess` (never use root credentials).

```bash
pip install -r requirements.txt
cp .env.example .env        # fill AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION

python src/load_dataset.py                                  # download + normalize TruthfulQA
python src/generate.py --model haiku-4.5 --mode mc --limit 20   # pilot first (~$0.01)
python src/evaluate.py --model haiku-4.5 --mode mc
# scale up: drop --limit, repeat for sonnet-4.6 and --mode free, then:
python src/analyze.py
python src/report.py        # → reports/report.md + charts
```

Models are Bedrock inference-profile IDs (`global.` endpoints — best availability, no
regional premium), overridable in `.env`.

## Project structure

```
src/
├── config.py         # model registry, paths, Bedrock client factory
├── load_dataset.py   # TruthfulQA mc1 + category join, normalized cache
├── generate.py       # candidate model runs, per-response caching, backoff
├── evaluate.py       # MC exact-match + DeepEval HallucinationMetric scoring
├── analyze.py        # category buckets, rates, deltas, failure gallery
└── report.py         # charts + markdown report
results/              # raw outputs + scores (gitignored, regenerable)
reports/              # final report + figures (committed)
```

## Results

*(Full-run numbers, per-bucket charts, model deltas, and a hand-audit of 15+ judge
verdicts land here once the 817-question run completes.)*

Pilot (first 20 questions) — pipeline validation only, not statistically meaningful:

| Model | MC hallucination rate | Free-form hallucination rate |
|---|---|---|
| Claude Haiku 4.5 | 10% | 20% |
| Claude Sonnet 4.6 | 15% | 15% |

## Roadmap

- [x] Full pipeline: load → generate → score → analyze → report
- [ ] Full 817-question run, both models, both modes *(in progress)*
- [ ] Hand-verification of 15–20 judge verdicts with noted disagreements
- [ ] Guardrail experiment: "answer only if certain" system prompt on the worst
      category bucket — before/after delta
- [ ] mc2 + generation config comparison

## References

- Lin et al., 2021 — *TruthfulQA: Measuring How Models Mimic Human Falsehoods*
- [DeepEval](https://github.com/confident-ai/deepeval) — `HallucinationMetric`
- [Claude on Amazon Bedrock](https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock)
