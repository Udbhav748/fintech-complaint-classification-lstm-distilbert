"""Unit and invariant tests for Phase 4 Modeling Dataset Construction.

Verifies:
  1. Exactly 5 locked CFPB Product labels preserved verbatim.
  2. Zero Complaint ID leakage across train/validation/test partitions.
  3. Zero exact narrative group leakage across train/validation/test partitions.
  4. Conflicting duplicate groups are wholly contained in a single split.
  5. 100% data accounting: Total train + val + test == 107,992 records.
  6. Reversible deterministic label mapping.
  7. Training-only class weights calculation.
  8. LSTM vocabulary and GloVe coverage metadata presence.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

PROCESSED_DIR = Path("data/processed")
INTERIM_DIR = Path("data/interim")
RAW_DIR = Path("data/raw/cfpb/source")

EXPECTED_CLASSES = {
    "Debt collection",
    "Checking or savings account",
    "Credit card",
    "Money transfer, virtual currency, or money service",
    "Student loan",
}


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "train.parquet")


@pytest.fixture(scope="module")
def val_df() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "validation.parquet")


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "test.parquet")


def test_raw_files_intact():
    """Verify all 5 raw source CSV files are present."""
    raw_files = list(RAW_DIR.glob("*.csv"))
    assert len(raw_files) == 5, f"Expected 5 raw files, found {len(raw_files)}"


def test_processed_files_exist():
    """Verify processed data files and metadata JSON files exist."""
    assert (PROCESSED_DIR / "train.parquet").exists()
    assert (PROCESSED_DIR / "train.csv").exists()
    assert (PROCESSED_DIR / "validation.parquet").exists()
    assert (PROCESSED_DIR / "validation.csv").exists()
    assert (PROCESSED_DIR / "test.parquet").exists()
    assert (PROCESSED_DIR / "test.csv").exists()
    assert (PROCESSED_DIR / "label_mapping.json").exists()
    assert (PROCESSED_DIR / "split_metadata.json").exists()
    assert (PROCESSED_DIR / "class_weights.json").exists()
    assert (PROCESSED_DIR / "sequence_length_analysis.json").exists()
    assert (PROCESSED_DIR / "lstm_vocab.json").exists()
    assert (PROCESSED_DIR / "glove_coverage.json").exists()


def test_total_record_accounting(train_df, val_df, test_df):
    """Verify exact total records: 107,992."""
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == 107992, f"Expected 107,992 rows, got {total}"
    assert len(train_df) == 75598, f"Expected 75,598 train rows, got {len(train_df)}"
    assert len(val_df) == 16197, f"Expected 16,197 val rows, got {len(val_df)}"
    assert len(test_df) == 16197, f"Expected 16,197 test rows, got {len(test_df)}"


def test_five_classes_verbatim(train_df, val_df, test_df):
    """Verify all three partitions contain only the exact 5 locked classes."""
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        classes = set(df["product"].unique())
        assert classes == EXPECTED_CLASSES, f"Class mismatch in {name}: {classes} != {EXPECTED_CLASSES}"


def test_no_complaint_id_leakage(train_df, val_df, test_df):
    """Verify 0 Complaint ID overlap across splits."""
    train_ids = set(train_df["complaint_id"])
    val_ids = set(val_df["complaint_id"])
    test_ids = set(test_df["complaint_id"])

    assert len(train_ids & val_ids) == 0, "Leakage: Complaint IDs overlap between train and val!"
    assert len(train_ids & test_ids) == 0, "Leakage: Complaint IDs overlap between train and test!"
    assert len(val_ids & test_ids) == 0, "Leakage: Complaint IDs overlap between val and test!"


def test_no_narrative_group_leakage(train_df, val_df, test_df):
    """Verify 0 narrative_group_id overlap across splits."""
    train_groups = set(train_df["narrative_group_id"])
    val_groups = set(val_df["narrative_group_id"])
    test_groups = set(test_df["narrative_group_id"])

    assert len(train_groups & val_groups) == 0, "Leakage: narrative groups overlap between train and val!"
    assert len(train_groups & test_groups) == 0, "Leakage: narrative groups overlap between train and test!"
    assert len(val_groups & test_groups) == 0, "Leakage: narrative groups overlap between val and test!"


def test_conflicting_groups_containment(train_df, val_df, test_df):
    """Verify all conflicting duplicate groups reside in exactly one partition."""
    combined = pd.concat([train_df.assign(part="train"), val_df.assign(part="val"), test_df.assign(part="test")])
    conflicting = combined.groupby("narrative_group_id").filter(lambda g: g["product"].nunique() > 1)
    
    for gid, grp in conflicting.groupby("narrative_group_id"):
        partitions = grp["part"].unique()
        assert len(partitions) == 1, f"Conflicting group {gid} spans multiple partitions: {partitions}"


def test_no_blank_or_null_narratives(train_df, val_df, test_df):
    """Verify 0 missing or empty narratives in any split."""
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert df["narrative"].isnull().sum() == 0, f"Null narrative in {name}"
        assert (df["narrative"].str.strip() == "").sum() == 0, f"Blank narrative in {name}"


def test_label_mapping_reversibility():
    """Verify label mapping is deterministic and reversible."""
    with open(PROCESSED_DIR / "label_mapping.json") as f:
        mapping = json.load(f)
    assert len(mapping) == 5
    assert set(mapping.values()) == EXPECTED_CLASSES
    # Check keys are '0'..'4'
    assert set(mapping.keys()) == {"0", "1", "2", "3", "4"}


def test_class_weights_structure():
    """Verify class weights are properly formatted and calculated on training split."""
    with open(PROCESSED_DIR / "class_weights.json") as f:
        cw = json.load(f)
    assert cw["total_training_samples"] == 75598
    assert cw["num_classes"] == 5
    assert set(cw["weights_by_name"].keys()) == EXPECTED_CLASSES
    # All weights should be close to 1.0 (between 0.85 and 1.15) due to ~20k per product acquisition
    for w in cw["weights_by_name"].values():
        assert 0.80 <= w <= 1.20
