"""Step 2 – Prepare the dataset.

Reads the original Kaggle files from data/ and writes clean files to data/processed/:

    data/train.csv  ->  processed/train.csv       (80%, used to train the model)
                    ->  processed/validation.csv  (20%, used to evaluate the model)
    data/test.csv   ->  processed/live.csv        (never seen in training; the simulator
                                                   sends these passengers to the API as live traffic)

Run from the project root:
    python -m training.prepare_data
"""

import argparse

from sklearn.model_selection import train_test_split

from skysat import schema
from skysat.data import DATA_DIR, PROCESSED_DIR, DataError, clean, load_raw

SEED = 42


def summarize(name, df):
    rate = df[schema.TARGET].mean() * 100
    print(f"  {name:<11} {len(df):>7,} rows   satisfied: {rate:4.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Clean and split the airline satisfaction dataset")
    parser.add_argument("--validation-size", type=float, default=0.2,
                        help="share of train.csv kept for validation (default 0.2)")
    args = parser.parse_args()

    try:
        train_raw = load_raw(DATA_DIR / "train.csv")
        test_raw = load_raw(DATA_DIR / "test.csv")
        train_all = clean(train_raw)
        live = clean(test_raw)
    except DataError as err:
        raise SystemExit(f"ERROR: {err}")

    # Stratified split keeps the satisfied / not satisfied ratio the same in both parts
    train, validation = train_test_split(
        train_all,
        test_size=args.validation_size,
        stratify=train_all[schema.TARGET],
        random_state=SEED,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    live.to_csv(PROCESSED_DIR / "live.csv", index=False)

    print(f"Clean data written to {PROCESSED_DIR.relative_to(DATA_DIR.parent)}/")
    summarize("train", train)
    summarize("validation", validation)
    summarize("live", live)
    filled = int(train_raw["Arrival Delay in Minutes"].isna().sum() + test_raw["Arrival Delay in Minutes"].isna().sum())
    print(f"  {filled} missing arrival delays filled with the departure delay")
    print(f"  {len(schema.FEATURES)} features: {len(schema.CATEGORICAL_FEATURES)} categorical, "
          f"{len(schema.NUMERIC_FEATURES)} numeric, {len(schema.RATING_FEATURES)} service ratings")


if __name__ == "__main__":
    main()
