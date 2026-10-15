"""Loading and cleaning the Airline Passenger Satisfaction dataset."""

from pathlib import Path

import pandas as pd

from skysat import schema

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


class DataError(Exception):
    """Raised when an input file does not look like the expected dataset."""


def load_raw(path: Path) -> pd.DataFrame:
    """Read a raw Kaggle CSV and check that every expected column is there."""
    if not path.exists():
        raise DataError(
            f"{path} not found. Download the dataset from Kaggle "
            "(see data/README.md) and put train.csv and test.csv into the data/ folder."
        )
    df = pd.read_csv(path)
    missing = [c for c in schema.RAW_TO_CLEAN if c not in df.columns]
    if missing:
        raise DataError(f"{path.name} is missing columns: {missing}")
    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw dataframe into the clean format used by the whole project.

    - keeps only the known columns and renames them to snake_case
    - fills the few missing arrival delays with the departure delay
    - adds the numeric target column `satisfied` (1 / 0)
    - checks categorical values and rating ranges
    """
    df = raw[list(schema.RAW_TO_CLEAN)].rename(columns=schema.RAW_TO_CLEAN)

    # ~0.3% of arrival delays are empty; the departure delay is the closest estimate
    df["arrival_delay_minutes"] = df["arrival_delay_minutes"].fillna(df["departure_delay_minutes"])

    for col in schema.NUMERIC_FEATURES + schema.RATING_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    for col, allowed in schema.CATEGORICAL_VALUES.items():
        unknown = sorted(set(df[col].unique()) - set(allowed))
        if unknown:
            raise DataError(f"Column {col} has unexpected values: {unknown}")

    ratings = df[schema.RATING_FEATURES]
    if ratings.min().min() < schema.RATING_MIN or ratings.max().max() > schema.RATING_MAX:
        raise DataError("Service ratings must be between 0 and 5")

    labels = set(df[schema.TARGET_RAW].unique())
    expected = {schema.POSITIVE_LABEL, "neutral or dissatisfied"}
    if not labels <= expected:
        raise DataError(f"Unexpected satisfaction labels: {sorted(labels - expected)}")

    df[schema.TARGET] = (df[schema.TARGET_RAW] == schema.POSITIVE_LABEL).astype(int)
    df = df.drop(columns=[schema.TARGET_RAW])

    if df.isna().any().any():
        raise DataError("Clean data still has missing values")

    return df[[schema.ID_COLUMN] + schema.FEATURES + [schema.TARGET]]
