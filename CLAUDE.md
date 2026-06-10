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
- Secrets live in `.env` (gitignored). Required: `OPENAI_API_KEY`.
  Load via `python-dotenv`. Never hardcode keys, never commit `.env`.

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
$PY src/generate.py --model gpt-4o-mini --split mc1        # collect outputs
$PY src/evaluate.py --run results/gpt-4o-mini_mc1.json     # score with DeepEval
$PY src/analyze.py --compare gpt-4o-mini gpt-4o            # category breakdown + delta
```
