"""The complaint box: real airline reviews in, a score, a recommendation and the main complaint out.

Data: Kaggle "joelljungstrom/128k-airline-reviews" (licence DbCL-1.0, scraped from airlinequality.com), the file
data/reviews/AirlineReviews.csv. It stays on the machine that downloaded it: never committed, never logged to MLflow,
never in a Docker image, never stored as text in Postgres.

The labels come from the reviewers themselves:
  score         OverallScore, 1-10
  recommended   yes / no
  complaints    the sub-ratings (1-5) tied for the lowest rated value if that value is <= 2, else ["none"]
                (a sub-rating of 0 or empty means "not rated": the CSV stores 0, see rated_values())

Splits (fixed seed, one function: split_reviews):
  held-out airlines  20% of the airlines, never in training or validation (the gate-on-new-airlines check)
  train              flown <= 2018, the other airlines
  validation         a random 10% of those <= 2018 rows, not in train: what the gate scores
  live               flown >= 2019, all airlines, sorted by date: the traffic after training (replay, drift, retraining)

The prompt (format_prompt) holds the airline, cabin, travel type, title and the review cut to 250 words. No date,
no label: the model must not be able to read the year or the answer.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

REVIEWS_FILE = Path("data/reviews/AirlineReviews.csv")
LAST_TRAIN_YEAR = 2018
HELD_OUT_AIRLINES = 0.20
VALIDATION_SHARE = 0.10
MAX_WORDS = 250
COMPLAINT_MAX = 2  # a lowest rating at or below this is a complaint

# CSV column -> the name used in the completion
SUB_RATINGS = {
    "SeatComfortRating": "seat",
    "ServiceRating": "service",
    "FoodRating": "food",
    "EntertainmentRating": "entertainment",
    "GroundServiceRating": "ground service",
    "WifiRating": "wifi",
    "ValueRating": "value",
}
COMPLAINT_NAMES = list(SUB_RATINGS.values())
NONE = "none"
COMPLAINT_CHOICES = COMPLAINT_NAMES + [NONE]

SYSTEM_PROMPT = ("You read an airline passenger's review and answer with exactly three lines: Score: <1-10>/10, "
                 "Recommended: <yes or no>, Complaint: <the main complaint, or none>.")


def rated_values(row) -> dict[str, int]:
    """Sub-rating name -> rating for the sub-ratings the reviewer filled in (0 and empty both mean not rated)."""
    out = {}
    for col, name in SUB_RATINGS.items():
        value = row.get(col)
        if value is not None and not pd.isna(value) and int(value) > 0:
            out[name] = int(value)
    return out


def complaints(row) -> list[str]:
    """The sub-ratings tied for the lowest rated value if that value is <= 2, else ['none']."""
    rated = rated_values(row)
    if not rated or min(rated.values()) > COMPLAINT_MAX:
        return [NONE]
    low = min(rated.values())
    return [name for name in COMPLAINT_NAMES if rated.get(name) == low]


def parse_date(values: pd.Series) -> pd.Series:
    """'November 2019' -> a timestamp; anything else (missing, odd text) -> NaT."""
    return pd.to_datetime(values, format="%B %Y", errors="coerce")


def clean_reviews(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a review, an OverallScore 1-10, a parseable DateFlown and Recommended yes/no; add the labels
    `score`, `recommended`, `complaints` (list) and `date`, `year`."""
    df = raw.copy()
    df["date"] = parse_date(df["DateFlown"])
    score = pd.to_numeric(df["OverallScore"], errors="coerce")
    recommended = df["Recommended"].astype(str).str.strip().str.lower()
    review = df["Review"].fillna("").astype(str).str.strip()
    keep = (review != "") & score.between(1, 10) & df["date"].notna() & recommended.isin(["yes", "no"])
    df = df[keep].copy()
    df["Review"] = review[keep]
    df["score"] = score[keep].round().astype(int)
    df["recommended"] = recommended[keep]
    df["year"] = df["date"].dt.year.astype(int)
    df["complaints"] = [complaints(r) for r in df.to_dict("records")]
    return df.reset_index(drop=True)


def load_reviews(path: Path | str = REVIEWS_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found. Run: kaggle datasets download joelljungstrom/128k-airline-reviews "
                         f"-p {path.parent} --unzip")
    return clean_reviews(pd.read_csv(path))


def split_reviews(df: pd.DataFrame, seed: int = 0) -> dict:
    """{'train', 'validation', 'live', 'held_out_airlines'}; see the module docstring."""
    airlines = np.array(sorted(df["AirlineName"].dropna().unique()))
    rng = np.random.default_rng(seed)
    n_held = max(1, round(HELD_OUT_AIRLINES * len(airlines)))
    held_out = sorted(rng.permutation(airlines)[:n_held].tolist())
    is_held = df["AirlineName"].isin(held_out)
    early = df[(df["year"] <= LAST_TRAIN_YEAR) & ~is_held]
    n_val = round(VALIDATION_SHARE * len(early))
    val_index = rng.permutation(early.index.to_numpy())[:n_val]
    validation = early.loc[np.sort(val_index)]
    train = early.drop(validation.index)
    live = df[df["year"] > LAST_TRAIN_YEAR].sort_values("date", kind="stable")
    return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True),
            "live": live.reset_index(drop=True), "held_out_airlines": held_out}


def split_year(live: pd.DataFrame, year: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, held-out) for retraining on one live year: a random 10% of that year's reviews is kept for the gate."""
    rows = live[live["year"] == year]
    if rows.empty:
        raise SystemExit(f"ERROR: no live reviews flown in {year}")
    order = np.random.default_rng(seed).permutation(rows.index.to_numpy())
    n_val = max(1, round(VALIDATION_SHARE * len(rows)))
    return rows.drop(order[:n_val]).reset_index(drop=True), rows.loc[np.sort(order[:n_val])].reset_index(drop=True)


def get_split(df: pd.DataFrame, split: str, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, gate rows) for --split train | live-YYYY."""
    if split == "train":
        parts = split_reviews(df, seed)
        return parts["train"], parts["validation"]
    m = re.fullmatch(r"live-(\d{4})", split)
    if not m:
        raise ValueError("split must be 'train' or 'live-YYYY'")
    live = split_reviews(df, seed)["live"]
    return split_year(live, int(m.group(1)), seed)


def _clean_text(value, default="unknown") -> str:
    text = " ".join(str(value).split()) if isinstance(value, str) else ""
    return text or default


def format_prompt(row) -> str:
    """What the model reads: airline, cabin, travel type, title, the review cut to 250 words. No date, no label."""
    words = _clean_text(row.get("Review"), "").split()
    review = " ".join(words[:MAX_WORDS]) + (" ..." if len(words) > MAX_WORDS else "")
    return (f"Airline: {_clean_text(row.get('AirlineName'))}\nCabin: {_clean_text(row.get('CabinType'))}\n"
            f"Travel type: {_clean_text(row.get('TravelType'))}\nTitle: {_clean_text(row.get('Title'), '')}\n"
            f"Review: {review}")


def format_completion(score: int, recommended: str, complaint: str) -> str:
    return f"Score: {int(score)}/10\nRecommended: {recommended}\nComplaint: {complaint}"


def parse_output(text: str) -> dict:
    """{'score': int 1-10 | None, 'recommended': 'yes'|'no'|None, 'complaint': a name or 'none' | None}.
    Tolerant of case, spacing, '7 out of 10', a trailing full stop; a malformed field is None, never a guess."""
    text = text or ""
    out = {"score": None, "recommended": None, "complaint": None}
    m = re.search(r"score\s*[:=\-]?\s*(\d{1,2})(?:\.\d+)?\s*(?:/|out of)?\s*(?:10)?", text, re.I)
    if m and 1 <= int(m.group(1)) <= 10:
        out["score"] = int(m.group(1))
    m = re.search(r"recommend\w*\s*[:=\-]?\s*(yes|no)\b", text, re.I)
    if m:
        out["recommended"] = m.group(1).lower()
    m = re.search(r"complaint\s*[:=\-]?\s*([^\n]*)", text, re.I)
    if m:
        name = re.sub(r"[\s_\-]+", " ", m.group(1).strip().strip(".").strip().lower())
        if name in COMPLAINT_CHOICES:
            out["complaint"] = name
    return out


def build_examples(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """n random reviews with `prompt` and `completion` (a tied complaint is picked at random, reproducibly)."""
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    picked = [str(rng.choice(c)) for c in sample["complaints"]]
    return sample.assign(
        prompt=[format_prompt(r) for r in sample.to_dict("records")],
        completion=[format_completion(s, r, c) for s, r, c in zip(sample["score"], sample["recommended"], picked)])
