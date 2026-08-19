"""Loading and deduplication for the CFPB complaint dataset."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/combined_complaints.parquet")
TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Product"
ID_COL = "Complaint ID"

LABELS = [
    "Checking or savings account",
    "Credit card",
    "Debt collection",
    "Money transfer, virtual currency, or money service",
    "Student loan",
]

SHORT_LABELS = {
    "Checking or savings account": "Checking/savings",
    "Credit card": "Credit card",
    "Debt collection": "Debt collection",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Student loan": "Student loan",
}


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - run src/data_processing/combine_data.py first")
    return pd.read_parquet(path)


def deduplicate(df: pd.DataFrame, text_col: str = TEXT_COL) -> pd.DataFrame:
    """Drop repeated narratives. Row order in the parquet file is fixed, so
    keeping the first occurrence is reproducible."""
    return df.drop_duplicates(subset=text_col, keep="first").reset_index(drop=True)


def duplicate_groups(df: pd.DataFrame, text_col: str = TEXT_COL) -> pd.Series:
    counts = df[text_col].value_counts()
    return counts[counts > 1]


def class_summary(labels: pd.Series) -> pd.DataFrame:
    counts = labels.value_counts()
    return pd.DataFrame({"n": counts, "pct": (100 * counts / len(labels)).round(2)})
