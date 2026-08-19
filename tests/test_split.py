"""Tests for split validation utilities and split generation invariants."""

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data import LABELS
from src.split import (
    SplitValidationError,
    create_stratified_split,
    load_splits,
    save_splits,
    validate_split,
)


class TestSplitValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        # Generate clean synthetic dataset with 5 classes and distinct narratives
        n_samples_per_class = 200
        rows = []
        uid = 1
        for label in LABELS:
            for i in range(n_samples_per_class):
                rows.append({
                    "Complaint ID": str(uid),
                    "Consumer complaint narrative": f"Unique narrative text for complaint {uid} regarding {label}",
                    "Product": label,
                })
                uid += 1
        self.clean_df = pd.DataFrame(rows)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_and_validate_stratified_split_success(self):
        train_idx, val_idx, test_idx = create_stratified_split(
            self.clean_df, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42
        )

        self.assertEqual(len(train_idx), 800)
        self.assertEqual(len(val_idx), 100)
        self.assertEqual(len(test_idx), 100)

        # Invariant checks
        validate_split(self.clean_df, train_idx, val_idx, test_idx)

    def test_save_and_load_splits(self):
        train_idx, val_idx, test_idx = create_stratified_split(self.clean_df, seed=42)
        save_splits(train_idx, val_idx, test_idx, output_dir=self.temp_dir)

        loaded_tr, loaded_v, loaded_te = load_splits(split_dir=self.temp_dir)
        np.testing.assert_array_equal(train_idx, loaded_tr)
        np.testing.assert_array_equal(val_idx, loaded_v)
        np.testing.assert_array_equal(test_idx, loaded_te)

    def test_fails_on_duplicate_safety_violation(self):
        # DataFrame containing unhandled duplicate narrative
        df_with_dups = self.clean_df.copy()
        df_with_dups.loc[1, "Consumer complaint narrative"] = df_with_dups.loc[0, "Consumer complaint narrative"]

        train_idx = np.arange(800)
        val_idx = np.arange(800, 900)
        test_idx = np.arange(900, 1000)

        with self.assertRaises(SplitValidationError) as ctx:
            validate_split(df_with_dups, train_idx, val_idx, test_idx)
        self.assertIn("Duplicate safety violated", str(ctx.exception))

    def test_fails_on_index_disjointness_violation(self):
        # Overlap between train and val
        train_idx = np.arange(800)
        val_idx = np.arange(795, 895)  # indices 795-799 overlap with train
        test_idx = np.arange(900, 1000)

        with self.assertRaises(SplitValidationError) as ctx:
            validate_split(self.clean_df, train_idx, val_idx, test_idx)
        self.assertIn("Index disjointness violated", str(ctx.exception))

    def test_fails_on_complete_coverage_violation(self):
        # Missing index 0
        train_idx = np.arange(1, 800)
        val_idx = np.arange(800, 900)
        test_idx = np.arange(900, 1000)

        with self.assertRaises(SplitValidationError) as ctx:
            validate_split(self.clean_df, train_idx, val_idx, test_idx)
        self.assertIn("Complete coverage violated", str(ctx.exception))

    def test_fails_on_cross_split_text_leakage(self):
        # Duplicate narrative injected across split boundaries
        df_leaked = self.clean_df.copy()
        df_leaked.loc[850, "Consumer complaint narrative"] = df_leaked.loc[10, "Consumer complaint narrative"]

        train_idx = np.arange(800)
        val_idx = np.arange(800, 900)
        test_idx = np.arange(900, 1000)

        with self.assertRaises(SplitValidationError) as ctx:
            validate_split(df_leaked, train_idx, val_idx, test_idx)
        self.assertIn("Cross-split text leakage detected", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
