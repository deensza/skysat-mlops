"""Data card: which data the project uses, described by aggregates only, logged to MLflow.

The DagsHub repo is public, so the data itself cannot go there: the competition data may only reach
participants (competition rules 2.4.b) and the original dataset's Kaggle licence ("other") does not grant
redistribution. What goes up instead, per file: source, licence, rows, columns, satisfied share, the schema
with value ranges, and the sha256 of the file, so anyone can download the data from its source and check
that their copy is byte-for-byte the one our models were trained on.

    python -m training.data_card            # or: make data-card  (logs to MLFLOW_TRACKING_URI)
    python -m training.data_card --no-mlflow  # only writes artifacts/data_card/
"""

import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd

from skysat.data import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT

EXPERIMENT = "skysat-data"
OUT_DIR = PROJECT_ROOT / "artifacts" / "data_card"
ORIGINAL_URL = "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction"
COMPETITION_URL = "https://www.kaggle.com/competitions/playground-series-s6e10/data"
TARGETS = ["satisfied", "satisfaction"]  # processed files / original and competition files
MAX_LISTED_VALUES = 10  # text columns with more distinct values only show the count

# (dataset, file name, where to read it, where it comes from, licence, what it is used for)
FILES = [
    ("original", "train.csv", DATA_DIR / "train.csv", ORIGINAL_URL,
     "Kaggle 'other' (modified from John D's customer-satisfaction dataset)", "make data -> processed/"),
    ("original", "test.csv", DATA_DIR / "test.csv", ORIGINAL_URL,
     "Kaggle 'other' (modified from John D's customer-satisfaction dataset)", "make data -> processed/live.csv"),
    ("processed", "train.csv", PROCESSED_DIR / "train.csv", "make data (80% of original train.csv, seed 42)",
     "as original", "training (CT, make train)"),
    ("processed", "validation.csv", PROCESSED_DIR / "validation.csv", "make data (20% of original train.csv, seed 42)",
     "as original", "validation, champion comparison"),
    ("processed", "live.csv", PROCESSED_DIR / "live.csv", "make data (original test.csv, cleaned)",
     "as original", "simulated live traffic, drift checks"),
    ("competition", "train.csv", DATA_DIR / "playground-series-s6e10.zip", COMPETITION_URL,
     "CC BY 4.0, participants only (rules 2.4)", "make submission (5-fold CV)"),
    ("competition", "test.csv", DATA_DIR / "playground-series-s6e10.zip", COMPETITION_URL,
     "CC BY 4.0, participants only (rules 2.4)", "make submission (predictions)"),
    ("competition", "sample_submission.csv", DATA_DIR / "playground-series-s6e10.zip", COMPETITION_URL,
     "CC BY 4.0, participants only (rules 2.4)", "submission format"),
]


def read_bytes(path: Path, name: str) -> bytes:
    """The file's bytes; for the competition, the member `name` of the zip (the hash Kaggle's files have)."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            return z.read(name)
    return path.read_bytes()


def satisfied_share(target: pd.Series) -> float:
    """Share of satisfied passengers, whatever the file calls them (1, True or "satisfied")."""
    return float(target.astype(str).str.lower().isin(["1", "true", "satisfied"]).mean())


def describe_column(s: pd.Series) -> dict:
    col = {"dtype": str(s.dtype), "missing": int(s.isna().sum()), "distinct": int(s.nunique())}
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        col.update(min=float(s.min()), max=float(s.max()), mean=round(float(s.mean()), 4))
    elif col["distinct"] <= MAX_LISTED_VALUES:
        col["values"] = {str(k): int(v) for k, v in s.value_counts().items()}
    return col


def describe_file(dataset: str, name: str, data: bytes, source: str, licence: str, used_for: str) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(io.BytesIO(data))
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # the original's unnamed index column
    # a label column has two values (sample_submission's "satisfaction" is a placeholder probability)
    target = next((t for t in TARGETS if t in df.columns and df[t].nunique() == 2), None)
    card = {"dataset": dataset, "file": name, "source": source, "licence": licence, "used_for": used_for,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "rows": len(df),
            "columns": len(df.columns), "target": target,
            "satisfied_share": round(satisfied_share(df[target]), 4) if target else None,
            "schema": {c: describe_column(df[c]) for c in df.columns}}
    return card, df


def build(files=FILES) -> tuple[list[dict], dict]:
    """Cards for every file that is on this PC (missing ones are reported and skipped)."""
    cards, frames = [], {}
    for dataset, name, path, source, licence, used_for in files:
        if not path.exists():
            print(f"  skipped {dataset}/{name}: {path} not found")
            continue
        card, df = describe_file(dataset, name, read_bytes(path, name), source, licence, used_for)
        cards.append(card)
        frames[f"{dataset}/{name}"] = df
    return cards, frames


def markdown(cards: list[dict]) -> str:
    lines = ["# SkySat data card", "",
             "Aggregates only: no rows of any dataset are stored here. Download the data from its source and "
             "compare `sha256sum` with the hashes below (competition: the files inside the zip).", "",
             "| Dataset | File | Rows | Columns | Satisfied | sha256 | Source | Licence | Used for |",
             "|---|---|---:|---:|---:|---|---|---|---|"]
    for c in cards:
        share = f"{c['satisfied_share']:.1%}" if c["satisfied_share"] is not None else "–"
        lines.append(f"| {c['dataset']} | {c['file']} | {c['rows']:,} | {c['columns']} | {share} | "
                     f"`{c['sha256'][:16]}…` | {c['source']} | {c['licence']} | {c['used_for']} |")
    for c in cards:
        lines += ["", f"## {c['dataset']}/{c['file']}", "", f"sha256 `{c['sha256']}`", "",
                  "| Column | Type | Missing | Distinct | Range / values |", "|---|---|---:|---:|---|"]
        for name, col in c["schema"].items():
            if "min" in col:
                detail = f"{col['min']:g} – {col['max']:g} (mean {col['mean']:g})"
            else:
                detail = ", ".join(f"{k} ({v:,})" for k, v in col.get("values", {}).items())
            lines.append(f"| {name} | {col['dtype']} | {col['missing']:,} | {col['distinct']:,} | {detail} |")
    return "\n".join(lines) + "\n"


def write_local(cards: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data_card.json").write_text(json.dumps(cards, indent=2))
    (OUT_DIR / "data_card.md").write_text(markdown(cards))
    return OUT_DIR


def log_run(cards: list[dict], frames: dict) -> None:
    """One run in experiment `skysat-data`: the card (md + json), counts as metrics, hashes as tags and one
    MLflow dataset entry per file (schema, row count, digest: MLflow stores no rows for these)."""
    import mlflow

    from training.track import git_commit
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5050")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="data-card", tags={"purpose": "data-card", "git_commit": git_commit()}):
        for c in cards:
            key = f"{c['dataset']}.{c['file'].removesuffix('.csv')}"
            mlflow.log_metrics({f"{key}.rows": c["rows"], f"{key}.columns": c["columns"]})
            if c["satisfied_share"] is not None:
                mlflow.log_metric(f"{key}.satisfied_share", c["satisfied_share"])
            mlflow.set_tag(f"{key}.sha256", c["sha256"])
            source = c["source"] if c["source"].startswith("http") else ORIGINAL_URL
            ds = mlflow.data.from_pandas(frames[f"{c['dataset']}/{c['file']}"], source=source,
                                         name=f"{c['dataset']}/{c['file']}", targets=c["target"])
            mlflow.log_input(ds, context=c["dataset"])
        mlflow.log_text(markdown(cards), "data_card.md")
        mlflow.log_dict(cards, "data_card.json")
    print(f"Logged to MLflow ({mlflow.get_tracking_uri()}, experiment {EXPERIMENT}): data_card.md, data_card.json")


def main():
    parser = argparse.ArgumentParser(description="Describe the datasets (aggregates and hashes only) and log them")
    parser.add_argument("--no-mlflow", action="store_true", help="only write artifacts/data_card/")
    args = parser.parse_args()

    cards, frames = build()
    if not cards:
        raise SystemExit("ERROR: no data found in data/. See data/README.md.")
    for c in cards:
        print(f"  {c['dataset'] + '/' + c['file']:<34} {c['rows']:>7,} rows  sha256 {c['sha256'][:12]}")
    print(f"Written to {write_local(cards).relative_to(PROJECT_ROOT)}/")
    if not args.no_mlflow:
        log_run(cards, frames)


if __name__ == "__main__":
    main()
