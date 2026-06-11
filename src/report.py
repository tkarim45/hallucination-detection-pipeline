"""Generate charts and reports/report.md from results/analysis CSVs.

Usage:
    python src/report.py
"""

import pandas as pd

from config import (
    ANALYSIS_DIR,
    FIGURES_DIR,
    JUDGE_MODEL_ID,
    MODELS,
    REPORTS_DIR,
    ensure_dirs,
)


def bucket_chart(by_bucket: pd.DataFrame, mode: str) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = by_bucket[by_bucket["mode"] == mode]
    if sub.empty:
        return None
    wide = sub.pivot(index="bucket", columns="model", values="rate").sort_index()
    ax = wide.plot.bar(figsize=(10, 5), rot=30)
    ax.set_ylabel("hallucination rate")
    ax.set_xlabel("")
    ax.set_title(f"Hallucination rate by category bucket — {mode} mode")
    ax.legend(title="model")
    fig_path = FIGURES_DIR / f"by_bucket_{mode}.png"
    ax.figure.tight_layout()
    ax.figure.savefig(fig_path, dpi=150)
    plt.close(ax.figure)
    return fig_path.name


def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = "\n".join(
        "| " + " | ".join(
            f"{v:.1%}" if isinstance(v, float) else str(v) for v in row
        ) + " |"
        for row in df.itertuples(index=False)
    )
    return "\n".join([header, sep, body])


def main() -> None:
    ensure_dirs()
    overall = pd.read_csv(ANALYSIS_DIR / "overall.csv")
    by_bucket = pd.read_csv(ANALYSIS_DIR / "by_bucket.csv")

    figures = {m: bucket_chart(by_bucket, m) for m in by_bucket["mode"].unique()}

    deltas_path = ANALYSIS_DIR / "deltas.csv"
    deltas = pd.read_csv(deltas_path) if deltas_path.exists() else None

    lines = [
        "# Hallucination Detection Report — TruthfulQA on AWS Bedrock",
        "",
        "## Methodology",
        "",
        "- **Dataset:** TruthfulQA `multiple_choice` (mc1) validation split, 817 "
        "questions; categories joined from the `generation` config.",
        "- **Candidate models (Bedrock):** "
        + ", ".join(f"`{k}` (`{v}`)" for k, v in MODELS.items())
        + ".",
        "- **Modes:** `mc` (lettered choices, exact-match scoring) and `free` "
        "(raw question, scored by DeepEval `HallucinationMetric` against the "
        f"reference correct answers with judge `{JUDGE_MODEL_ID}`).",
        "- **Determinism:** temperature=0, pinned Bedrock model IDs, raw outputs "
        "cached before scoring.",
        "",
        "## Headline numbers",
        "",
        md_table(overall),
        "",
        "## Hallucination rate by category bucket",
        "",
    ]
    for mode, fig in figures.items():
        if fig:
            lines += [f"### {mode} mode", "", f"![by bucket — {mode}](figures/{fig})", ""]
    if deltas is not None:
        lines += ["## Model deltas per bucket", "", md_table(deltas.round(3)), ""]

    gallery_files = sorted(ANALYSIS_DIR.glob("failure_gallery_*.csv"))
    if gallery_files:
        gallery = pd.read_csv(gallery_files[0])
        lines += ["## Failure gallery (all models wrong)", ""]
        for qid, grp in gallery.groupby("id"):
            cat = grp["category"].iloc[0]
            lines.append(f"- **Q{qid}** ({cat})")
            for _, r in grp.iterrows():
                lines.append(f"  - `{r['model']}`: {str(r['output'])[:160]}")
        lines.append("")

    lines += [
        "## Limitations",
        "",
        "- LLM-judge verdicts are imperfect — sanity-check a sample by hand and "
        "record disagreements.",
        "- MC accuracy and free-form hallucination measure different behaviors; "
        "expect a gap between modes.",
        "- Single run per (model, mode); no variance estimate.",
        "",
        "## Recommendation",
        "",
        "- For buckets with the highest rates (typically health/legal), add a "
        "system-prompt guardrail ('answer only if certain, otherwise say you "
        "don't know') and re-measure before/after.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
