"""Quality gate for the complaint-box LLM, measured against what the reviewers themselves wrote.

For every review the model's three lines are parsed (llm/reviews.py parse_output) and compared with the labels:
  score_mae             mean |predicted - true OverallScore| (a malformed score counts as an error of 9)
  score_within1         share of reviews whose score is within 1 point
  recommended_accuracy  share with the right yes/no
  complaint_accuracy    share whose complaint is in the review's tied list (the lowest sub-ratings, or "none")
A malformed field is wrong. combined is one number to compare two adapters on the same reviews.

The limits come from measured runs (see the constants).
"""

import pandas as pd

from llm.reviews import parse_output

MAX_ERROR = 9  # the error counted for a malformed score: the widest possible on a 1-10 scale

# Set from the first real runs (Qwen2.5-1.5B + LoRA r=16, 2,000 reviews, RTX 4070, 2026-11-13):
#   champion v1 on normal reviews (2018 validation, 2019, 2020, 2022): MAE 0.75-0.91, recommended 0.94-0.96
#   v1 under the harsh survey (2019):                                    MAE 1.03-1.18, recommended 0.90-0.92
#   v2 retrained on harsh 2019:                                          MAE 0.41,      recommended 0.955
#   constant answers: MAE 1.2-3.2, always "no" 0.57-0.85, always "ground service" complaint 0.43-0.64
MAE_MAX = 1.0
RECOMMENDED_MIN = 0.90
COMPLAINT_MIN = 0.65


def score(rows: pd.DataFrame, texts: list[str]) -> dict:
    """The four measures and `combined` for rows (columns score, recommended, complaints) and the model's texts."""
    if len(rows) != len(texts) or not len(rows):
        raise ValueError("need one text per review, and at least one review")
    errors, rec_ok, comp_ok = [], [], []
    for (_, row), text in zip(rows.iterrows(), texts, strict=True):
        p = parse_output(text)
        errors.append(MAX_ERROR if p["score"] is None else abs(p["score"] - int(row["score"])))
        rec_ok.append(p["recommended"] == row["recommended"])
        comp_ok.append(p["complaint"] is not None and p["complaint"] in list(row["complaints"]))
    n = len(errors)
    m = {"score_mae": sum(errors) / n,
         "score_within1": sum(e <= 1 for e in errors) / n,
         "recommended_accuracy": sum(rec_ok) / n,
         "complaint_accuracy": sum(comp_ok) / n,
         "n": n}
    m["combined"] = combined(m)
    return m


def combined(m: dict) -> float:
    """mean(1 - mae/9, recommended_accuracy, complaint_accuracy): 1.0 is perfect."""
    return (1 - m["score_mae"] / MAX_ERROR + m["recommended_accuracy"] + m["complaint_accuracy"]) / 3


def passes(m: dict) -> bool:
    return (m["score_mae"] <= MAE_MAX and m["recommended_accuracy"] >= RECOMMENDED_MIN
            and m["complaint_accuracy"] >= COMPLAINT_MIN)


def report(m: dict) -> str:
    rows = [
        ("score_mae", f"{m['score_mae']:.2f}", f"<= {MAE_MAX}"),
        ("score_within1", f"{m['score_within1']:.3f}", ""),
        ("recommended_accuracy", f"{m['recommended_accuracy']:.3f}", f">= {RECOMMENDED_MIN}"),
        ("complaint_accuracy", f"{m['complaint_accuracy']:.3f}", f">= {COMPLAINT_MIN}"),
        ("combined", f"{m['combined']:.3f}", ""),
    ]
    lines = [f"  {name:<21} {value:>6}  {limit}".rstrip() for name, value, limit in rows]
    return "\n".join(lines + [f"  n = {m['n']} reviews; {'PASSES' if passes(m) else 'FAILS'}"])
