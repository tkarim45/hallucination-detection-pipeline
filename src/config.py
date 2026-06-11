"""Shared configuration: model registry, paths, Bedrock client factory."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
SCORES_DIR = RESULTS_DIR / "scores"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATASET_CACHE = RESULTS_DIR / "dataset.json"

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

# Bedrock inference-profile IDs. "global." routes for max availability with no
# regional premium; swap to "us." prefix if your account lacks global profiles.
MODELS = {
    "haiku-4.5": os.getenv(
        "BEDROCK_MODEL_A", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    ),
    "sonnet-4.6": os.getenv("BEDROCK_MODEL_B", "global.anthropic.claude-sonnet-4-6"),
}

# Judge is deliberately a stronger model than both candidates to reduce
# judge-capability bias in the hallucination verdicts.
JUDGE_MODEL_ID = os.getenv("BEDROCK_JUDGE_MODEL", "global.anthropic.claude-opus-4-6-v1")

GENERATION_TEMPERATURE = 0.0
MC_MAX_TOKENS = 16
FREE_MAX_TOKENS = 256


def bedrock_client():
    """AnthropicBedrock client using AWS_* env vars (loaded from .env)."""
    from anthropic import AnthropicBedrock

    return AnthropicBedrock(aws_region=AWS_REGION)


def ensure_dirs():
    for d in (RAW_DIR, SCORES_DIR, ANALYSIS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
