"""Standardized raw CFPB data loader and schema validator for Phase 4.

Loads all five raw CFPB source CSV files, validates schema and class integrity,
standardizes column names, and produces a clean intermediate DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Column mapping from CFPB raw headers to standardized snake_case
RAW_TO_STANDARDIZED_COLUMNS: dict[str, str] = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company response to consumer": "company_response_to_consumer",
    "Timely response?": "timely_response",
    "Company public response": "company_public_response",
}

EXPECTED_RAW_COLUMNS = list(RAW_TO_STANDARDIZED_COLUMNS.keys())

LOCKED_PRODUCT_LABELS = [
    "Debt collection",
    "Checking or savings account",
    "Credit card",
    "Money transfer, virtual currency, or money service",
    "Student loan",
]


def load_and_validate_raw_data(
    source_dir: Path | str,
    expected_products: list[str] | None = None,
) -> pd.DataFrame:
    """Load all raw CSV files from source_dir, validate schema and labels,
    and return a standardized DataFrame.

    Raises:
        FileNotFoundError: If source_dir does not exist or has no CSV files.
        ValueError: If schema mismatch, unexpected product, or data integrity issue.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Raw source directory not found: {source_path}")

    csv_files = sorted(source_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {source_path}")

    expected_labels = set(expected_products or LOCKED_PRODUCT_LABELS)
    loaded_dfs: list[pd.DataFrame] = []

    logger.info("Found %d raw CSV files in %s", len(csv_files), source_path)

    for csv_file in csv_files:
        logger.info("Loading raw file: %s (size: %.2f MB)", csv_file.name, csv_file.stat().st_size / (1024 * 1024))
        df = pd.read_csv(csv_file, dtype=str, keep_default_na=False)

        # Validate schema
        missing_cols = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"File {csv_file.name} missing expected columns: {missing_cols}")

        # Check native Product column
        file_products = set(df["Product"].unique())
        unexpected = file_products - expected_labels
        if unexpected:
            raise ValueError(
                f"File {csv_file.name} contains unexpected Product labels: {unexpected}. "
                f"Expected only: {expected_labels}"
            )

        loaded_dfs.append(df)

    combined_df = pd.concat(loaded_dfs, ignore_index=True)
    logger.info("Combined raw records: %d rows, %d columns", len(combined_df), len(combined_df.columns))

    # Standardize column names
    standardized_df = combined_df[EXPECTED_RAW_COLUMNS].rename(columns=RAW_TO_STANDARDIZED_COLUMNS)

    # Convert Complaint ID to clean string
    standardized_df["complaint_id"] = standardized_df["complaint_id"].astype(str).str.strip()

    # Strip leading/trailing whitespace from text and labels
    standardized_df["product"] = standardized_df["product"].astype(str).str.strip()
    standardized_df["narrative"] = standardized_df["narrative"].astype(str).str.strip()

    # Enforce basic integrity invariants
    if standardized_df["complaint_id"].duplicated().any():
        dup_ids = standardized_df[standardized_df["complaint_id"].duplicated()]["complaint_id"].tolist()[:5]
        raise ValueError(f"Found duplicate Complaint IDs in raw data! Examples: {dup_ids}")

    empty_narratives = (standardized_df["narrative"] == "").sum()
    if empty_narratives > 0:
        raise ValueError(f"Found {empty_narratives} empty complaint narratives in raw data!")

    # Verify all expected products are present
    found_products = set(standardized_df["product"].unique())
    if found_products != expected_labels:
        raise ValueError(f"Product mismatch! Found {found_products}, expected {expected_labels}")

    return standardized_df
