# CLAUDE.md — Hallucination Detection Pipeline

## What this repo is

An LLM evaluation project: a pipeline that scores LLM outputs for factual accuracy
against the TruthfulQA benchmark (817 labeled questions, 38 topic categories), then
breaks hallucination rates down per topic category and compares two models side by
side. See `README.md` for the full project specification and build plan.

## Environment

- Python env: conda `personal` env (`~/miniconda3/envs/personal/bin/python`).
  NEVER use the conda `base` env or system Python.
- Install deps: `~/miniconda3/envs/personal/bin/pip install -r requirements.txt`
- Secrets live in `.env` (gitignored). Required: `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` (IAM user with Bedrock access — never root keys).
  Load via `python-dotenv`. Never hardcode keys, never commit `.env`.
- Models run on **AWS Bedrock** via the `anthropic[bedrock]` SDK (`AnthropicBedrock`
  client). DeepEval judge uses `deepeval.models.AmazonBedrockModel` (boto3 Converse).
  Model IDs are Bedrock inference-profile IDs (`global.anthropic.*`) — see `src/config.py`.

## Project structure (target layout)

```
.
├── CLAUDE.md
├── README.md                  # full project spec — read this first
├── requirements.txt
├── .env.example
├── src/
│   ├── load_dataset.py        # TruthfulQA loading + normalization
│   ├── generate.py            # run candidate models over the questions
│   ├── evaluate.py            # DeepEval HallucinationMetric scoring
│   ├── analyze.py             # per-category breakdown, model comparison
│   └── report.py              # charts + markdown report generation
├── results/                   # raw model outputs + scores (JSON/CSV, gitignored if large)
└── reports/                   # final analysis report + figures (committed)
```

## Conventions

- Every run must be reproducible: pin model names (exact API model IDs), pin
  dataset revision, set temperature=0 for generation, save raw outputs before scoring.
- Cache API responses to disk (JSON) so re-running analysis never re-spends API calls.
- Scores and analysis are derived artifacts — always regenerable from `results/` raw data.
- Keep cost in mind: batch requests, run small samples (20–50 questions) end-to-end
  first, then scale to the full 817.
- Tests/scratch scripts Claude writes for itself → run with `claude` env, not `personal`.

## Key commands (once built)

```bash
PY=~/miniconda3/envs/personal/bin/python
$PY src/load_dataset.py                                    # download + normalize TruthfulQA
$PY src/generate.py --model haiku-4.5 --mode mc --limit 20 # collect outputs (cached)
$PY src/generate.py --model sonnet-4.6 --mode free         # free-form, full 817
$PY src/evaluate.py --model haiku-4.5 --mode mc            # score (MC exact-match)
$PY src/evaluate.py --model haiku-4.5 --mode free          # score (DeepEval judge)
$PY src/analyze.py                                         # category breakdown + deltas
$PY src/report.py                                          # charts + reports/report.md
```
