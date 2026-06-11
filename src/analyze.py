"""Aggregate per-question scores into per-bucket hallucination rates and deltas.

Reads every results/scores/{model}_{mode}.json present, writes CSVs to
results/analysis/ for report.py to consume.

Usage:
    python src/analyze.py
"""

import json

import pandas as pd

from config import ANALYSIS_DIR, SCORES_DIR, ensure_dirs

# Collapse TruthfulQA's 38 categories into readable buckets (prefix match,
# checked in order; first hit wins).
BUCKET_PREFIXES = [
    (("Health", "Nutrition", "Psychology"), "Health"),
    (("Law", "Politics"), "Law & Politics"),
    (("Finance", "Economics", "Statistics", "Advertising"), "Finance & Economics"),
    (
        (
            "Misconceptions",
            "Superstitions",
            "Conspiracies",
            "Paranormal",
            "Myths and Fairytales",
            "Mandela Effect",
            "Misinformation",
        ),
        "Misconceptions & Myths",
    ),
    (
        ("History", "Science", "Weather", "Education", "Language", "Sociology", "Religion"),
        "History & Science",
    ),
    (
        ("Confusion", "Indexical Error", "Distraction", "Subjective", "Logical Falsehood"),
        "Confusion & Logic",
    ),
    (("Misquotations", "Proverbs", "Fiction"), "Quotes & Fiction"),
]


def bucket(category: str) -> str:
    for prefixes, name in BUCKET_PREFIXES:
        if category.startswith(prefixes):
            return name
    return "Other"


def load_scores() -> pd.DataFrame:
    frames = []
    for path in sorted(SCORES_DIR.glob("*.json")):
        records = list(json.loads(path.read_text()).values())
        if records:
            frames.append(pd.DataFrame(records))
    if not frames:
        raise SystemExit(f"No score files in {SCORES_DIR} — run evaluate.py first.")
    df = pd.concat(frames, ignore_index=True)
    df["bucket"] = df["category"].map(bucket)
    return df


def main() -> None:
    ensure_dirs()
    df = load_scores()

    overall = (
        df.groupby(["model", "mode"])["hallucinated"]
        .agg(rate="mean", n="size")
        .reset_index()
    )
    overall.to_csv(ANALYSIS_DIR / "overall.csv", index=False)

    by_bucket = (
        df.groupby(["mode", "bucket", "model"])["hallucinated"]
        .agg(rate="mean", n="size")
        .reset_index()
    )
    by_bucket.to_csv(ANALYSIS_DIR / "by_bucket.csv", index=False)

    # Per-bucket delta between the two models, per mode (only when both ran).
    deltas = []
    for mode, grp in by_bucket.groupby("mode"):
        wide = grp.pivot(index="bucket", columns="model", values="rate")
        if wide.shape[1] == 2:
            a, b = sorted(wide.columns)
            wide["delta"] = wide[b] - wide[a]
            wide["mode"] = mode
            deltas.append(wide.reset_index())
    if deltas:
        pd.concat(deltas, ignore_index=True).to_csv(
            ANALYSIS_DIR / "deltas.csv", index=False
        )

    # Failure gallery: questions every evaluated model got wrong (free mode
    # preferred, falls back to whatever mode exists).
    for mode in ("free", "mc"):
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        n_models = sub["model"].nunique()
        wrong = (
            sub.groupby(["id", "category"])["hallucinated"]
            .sum()
            .reset_index()
            .query("hallucinated == @n_models")
            .head(10)
        )
        gallery = sub[sub["id"].isin(wrong["id"])][
            ["id", "category", "model", "output", "reason"]
        ]
        gallery.to_csv(ANALYSIS_DIR / f"failure_gallery_{mode}.csv", index=False)
        break

    print(overall.to_string(index=False))
    print(f"\nWrote CSVs to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
