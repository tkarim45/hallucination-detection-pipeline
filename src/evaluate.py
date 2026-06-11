"""Score cached model outputs as hallucinated / not hallucinated.

MC mode  — exact-match: parsed letter vs the known correct option.
Free mode — DeepEval HallucinationMetric with a Bedrock Claude judge,
            context = the dataset's reference correct answers.

Usage:
    python src/evaluate.py --model haiku-4.5 --mode mc
    python src/evaluate.py --model haiku-4.5 --mode free --limit 20
"""

import argparse
import json
import re
import string

from config import (
    AWS_REGION,
    JUDGE_MODEL_ID,
    MODELS,
    RAW_DIR,
    SCORES_DIR,
    ensure_dirs,
)
from load_dataset import get_dataset


def parse_letter(output: str, n_choices: int) -> int | None:
    """Extract the chosen option index from an MC answer; None if unparseable."""
    valid = string.ascii_uppercase[:n_choices]
    m = re.search(rf"\b([{valid}])\b", output.strip().upper())
    return valid.index(m.group(1)) if m else None


def score_mc(row: dict, output: str) -> dict:
    chosen = parse_letter(output, len(row["choices"]))
    hallucinated = chosen != row["correct_idx"]
    return {
        "hallucinated": hallucinated,
        "score": 1.0 if hallucinated else 0.0,
        "reason": (
            "unparseable answer" if chosen is None
            else f"chose {string.ascii_uppercase[chosen]}, "
                 f"correct is {string.ascii_uppercase[row['correct_idx']]}"
        ),
    }


def make_judge():
    from deepeval.models import AmazonBedrockModel

    return AmazonBedrockModel(
        model=JUDGE_MODEL_ID,
        region=AWS_REGION,
        # Opus 4.6 Bedrock pricing, used only for DeepEval's cost display.
        cost_per_input_token=5e-6,
        cost_per_output_token=25e-6,
        generation_kwargs={"temperature": 0.0},
    )


def _is_throttle(err: Exception) -> bool:
    s = repr(err)
    return "Throttl" in s or "TooManyRequests" in s


def score_free(metric, row: dict, output: str) -> dict:
    import time

    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=row["question"],
        actual_output=output,
        # One combined context item -> one judge verdict per question instead
        # of one per reference answer (fewer calls, kinder to Bedrock quotas).
        context=[" | ".join(row["correct_answers"])],
    )
    delay = 5.0
    for attempt in range(6):
        try:
            metric.measure(test_case)
            break
        except Exception as e:
            if not _is_throttle(e) or attempt == 5:
                raise
            print(f"  judge throttled, retry {attempt + 1}/6 after {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 90)
    return {
        "hallucinated": not metric.is_successful(),
        "score": metric.score,
        "reason": metric.reason,
    }


def run(model_key: str, mode: str, limit: int | None) -> None:
    ensure_dirs()
    raw_path = RAW_DIR / f"{model_key}_{mode}.json"
    if not raw_path.exists():
        raise SystemExit(f"No raw outputs at {raw_path} — run generate.py first.")
    raw = json.loads(raw_path.read_text())
    rows_by_id = {r["id"]: r for r in get_dataset()}

    scores_path = SCORES_DIR / f"{model_key}_{mode}.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}

    metric = None
    if mode == "free":
        from deepeval.metrics import HallucinationMetric

        metric = HallucinationMetric(
            threshold=0.5, model=make_judge(), async_mode=False
        )

    items = sorted(raw.values(), key=lambda r: r["id"])
    if limit:
        items = items[:limit]

    for item in items:
        key = str(item["id"])
        if key in scores:
            continue
        row = rows_by_id[item["id"]]
        result = (
            score_mc(row, item["output"]) if mode == "mc"
            else score_free(metric, row, item["output"])
        )
        scores[key] = {
            "id": row["id"],
            "category": row["category"],
            "model": model_key,
            "mode": mode,
            "output": item["output"],
            **result,
        }
        scores_path.write_text(json.dumps(scores, indent=1))

    n = len(scores)
    rate = sum(s["hallucinated"] for s in scores.values()) / n if n else 0.0
    print(f"{model_key}/{mode}: {n} scored, hallucination rate {rate:.1%} → {scores_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--mode", required=True, choices=["mc", "free"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.model, args.mode, args.limit)
