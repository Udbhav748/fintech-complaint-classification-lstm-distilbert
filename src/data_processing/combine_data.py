"""Data combination script for CFPB complaint dataset.

Combines the 5 raw product category CSV files into a unified master dataset
while validating schema integrity and ensuring zero data loss.
"""

from pathlib import Path
import pandas as pd


def combine_raw_datasets(
    raw_dir: Path = Path("data/raw/cfpb/source"),
    output_csv: Path = Path("data/combined_complaints.csv"),
    output_parquet: Path = Path("data/combined_complaints.parquet"),
) -> pd.DataFrame:
    raw_files = sorted(list(raw_dir.glob("*.csv")))
    if not raw_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    dfs = []
    for filepath in raw_files:
        df = pd.read_csv(filepath, dtype=str)
        dfs.append(df)

    master_df = pd.concat(dfs, ignore_index=True)

    # Invariant checks
    assert len(master_df) == 107992, f"Expected 107,992 rows, got {len(master_df)}"
    assert master_df["Complaint ID"].nunique() == len(master_df), "Duplicate Complaint IDs detected"
    assert master_df["Consumer complaint narrative"].notna().all(), "Null narratives detected"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_csv, index=False)
    master_df.to_parquet(output_parquet, index=False)

    return master_df


if __name__ == "__main__":
    df = combine_raw_datasets()
    print(f"Successfully combined {len(df):,} records into unified dataset.")
