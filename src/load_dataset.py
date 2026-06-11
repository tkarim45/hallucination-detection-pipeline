"""Load TruthfulQA (mc1 + generation configs), join categories, normalize, cache.

Normalized row schema:
    {id, question, category, choices, correct_idx,
     best_answer, correct_answers, incorrect_answers}
"""

import json

from config import DATASET_CACHE, RESULTS_DIR


def build_dataset() -> list[dict]:
    from datasets import load_dataset

    mc = load_dataset("truthful_qa", "multiple_choice")["validation"]
    gen = load_dataset("truthful_qa", "generation")["validation"]

    # The MC config has no category field — join it back from the generation
    # config by question text. One question differs in whitespace between the
    # two configs, so the join key is whitespace/case-normalized.
    def norm(q: str) -> str:
        return " ".join(q.split()).lower()

    meta_by_question = {
        norm(row["question"]): {
            "category": row["category"],
            "best_answer": row["best_answer"],
            "correct_answers": row["correct_answers"],
            "incorrect_answers": row["incorrect_answers"],
        }
        for row in gen
    }

    rows = []
    for i, row in enumerate(mc):
        meta = meta_by_question.get(norm(row["question"]))
        if meta is None:
            raise ValueError(f"No generation-config match for: {row['question']!r}")
        labels = row["mc1_targets"]["labels"]
        rows.append(
            {
                "id": i,
                "question": row["question"],
                "category": meta["category"],
                "choices": row["mc1_targets"]["choices"],
                "correct_idx": labels.index(1),
                "best_answer": meta["best_answer"],
                "correct_answers": meta["correct_answers"],
                "incorrect_answers": meta["incorrect_answers"],
            }
        )
    return rows


def get_dataset() -> list[dict]:
    """Return the normalized dataset, building and caching it on first call."""
    if DATASET_CACHE.exists():
        return json.loads(DATASET_CACHE.read_text())
    rows = build_dataset()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_CACHE.write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    rows = get_dataset()
    print(f"{len(rows)} questions, {len({r['category'] for r in rows})} categories\n")
    for r in rows[:5]:
        print(f"[{r['id']}] ({r['category']}) {r['question']}")
        print(f"    correct: {r['choices'][r['correct_idx']]}")
