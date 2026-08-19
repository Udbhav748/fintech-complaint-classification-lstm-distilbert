"""Unit and integration tests for dataset fingerprinting and experiment metadata tracking."""

import datetime
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.fingerprint import (
    DatasetFingerprint,
    DatasetFingerprintMismatchError,
    SplitFingerprint,
    SplitFingerprintMismatchError,
    create_dataset_fingerprint,
    create_split_fingerprint,
    verify_dataset_fingerprint,
    verify_split_fingerprint,
)
from src.metadata import (
    EnvironmentMetadata,
    EvaluationMetrics,
    ExperimentRecord,
    ExperimentTracker,
)


class TestFingerprint(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_df = pd.DataFrame({
            "Complaint ID": ["101", "102", "103", "104", "105"],
            "Consumer complaint narrative": [
                "Problem with credit card fees",
                "Unauthorized money transfer",
                "Debt collection harassment",
                "Checking account closed unexpectedly",
                "Student loan repayment issue",
            ],
            "Product": [
                "Credit card",
                "Money transfer, virtual currency, or money service",
                "Debt collection",
                "Checking or savings account",
                "Student loan",
            ],
        })
        self.sample_parquet = self.temp_dir / "sample.parquet"
        self.sample_df.to_parquet(self.sample_parquet, index=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_and_save_dataset_fingerprint(self):
        fp = create_dataset_fingerprint(self.sample_df)
        self.assertEqual(fp.total_rows, 5)
        self.assertEqual(fp.num_columns, 3)
        self.assertEqual(fp.unique_narratives, 5)
        self.assertEqual(fp.duplicate_narratives_count, 0)
        self.assertTrue(len(fp.content_sha256) == 64)

        manifest_path = self.temp_dir / "fingerprint.json"
        fp.save(manifest_path)
        self.assertTrue(manifest_path.exists())

        loaded_fp = DatasetFingerprint.load(manifest_path)
        self.assertEqual(fp.content_sha256, loaded_fp.content_sha256)
        self.assertEqual(fp.total_rows, loaded_fp.total_rows)

    def test_verify_dataset_fingerprint_success(self):
        fp = create_dataset_fingerprint(self.sample_df)
        is_valid, mismatches = verify_dataset_fingerprint(self.sample_df, fp)
        self.assertTrue(is_valid)
        self.assertEqual(len(mismatches), 0)

    def test_verify_dataset_fingerprint_mismatch(self):
        fp = create_dataset_fingerprint(self.sample_df)
        # Corrupt data
        corrupted_df = self.sample_df.copy()
        corrupted_df.loc[0, "Consumer complaint narrative"] = "Altered narrative"

        with self.assertRaises(DatasetFingerprintMismatchError):
            verify_dataset_fingerprint(corrupted_df, fp, raise_on_mismatch=True)

        is_valid, mismatches = verify_dataset_fingerprint(corrupted_df, fp, raise_on_mismatch=False)
        self.assertFalse(is_valid)
        self.assertTrue(any("Content SHA-256 mismatch" in m for m in mismatches))

    def test_split_fingerprint_and_integrity(self):
        train_idx = np.array([0, 1, 2], dtype=np.int64)
        val_idx = np.array([3], dtype=np.int64)
        test_idx = np.array([4], dtype=np.int64)

        split_fp = create_split_fingerprint(self.sample_df, train_idx, val_idx, test_idx)
        self.assertEqual(split_fp.train_count, 3)
        self.assertEqual(split_fp.val_count, 1)
        self.assertEqual(split_fp.test_count, 1)
        self.assertTrue(split_fp.disjoint_indices_verified)
        self.assertTrue(split_fp.zero_duplicate_cross_split_verified)

        is_valid, mismatches = verify_split_fingerprint(self.sample_df, train_idx, val_idx, test_idx, split_fp)
        self.assertTrue(is_valid)
        self.assertEqual(len(mismatches), 0)

    def test_split_overlap_detection(self):
        # Overlapping indices
        train_idx = np.array([0, 1, 2], dtype=np.int64)
        val_idx = np.array([2, 3], dtype=np.int64)  # index 2 is duplicated
        test_idx = np.array([4], dtype=np.int64)

        with self.assertRaises(SplitFingerprintMismatchError):
            create_split_fingerprint(self.sample_df, train_idx, val_idx, test_idx)


class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tracker = ExperimentTracker(results_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_environment_capture(self):
        env = EnvironmentMetadata.capture()
        self.assertTrue(len(env.python_version) > 0)
        self.assertIn("pandas", env.package_versions)
        self.assertIn("numpy", env.package_versions)

    def test_log_run_and_comparison_table(self):
        start_time = datetime.datetime(2026, 8, 20, 10, 0, 0, tzinfo=datetime.timezone.utc)
        end_time = datetime.datetime(2026, 8, 20, 10, 5, 30, tzinfo=datetime.timezone.utc)

        metrics_m0_s42 = EvaluationMetrics(
            train_loss=0.45,
            val_loss=0.50,
            val_accuracy=0.8450,
            val_macro_f1=0.8420,
            val_macro_precision=0.8430,
            val_macro_recall=0.8410,
        )

        record1 = self.tracker.log_run(
            model_name="M0",
            seed=42,
            started_at=start_time,
            ended_at=end_time,
            epochs_trained=10,
            best_epoch=7,
            trainable_params=150000,
            total_params=150000,
            hyperparameters={"lr": 0.001, "batch_size": 64, "embedding_dim": 100},
            dataset_content_sha256="dummy_hash_123",
            split_indices_sha256={"train": "hash_tr", "val": "hash_v", "test": "hash_te"},
            metrics=metrics_m0_s42,
        )

        # Check JSON created
        json_path = self.temp_dir / "metadata" / f"{record1.run_id}.json"
        self.assertTrue(json_path.exists())

        # Check CSV created
        runs_csv = self.temp_dir / "runs.csv"
        self.assertTrue(runs_csv.exists())
        runs_df = pd.read_csv(runs_csv)
        self.assertEqual(len(runs_df), 1)
        self.assertEqual(runs_df.iloc[0]["model_name"], "M0")
        self.assertEqual(runs_df.iloc[0]["best_val_macro_f1"], 0.8420)

        # Log a second run for M1
        metrics_m1_s42 = EvaluationMetrics(
            train_loss=0.38,
            val_loss=0.42,
            val_accuracy=0.8610,
            val_macro_f1=0.8580,
            val_macro_precision=0.8590,
            val_macro_recall=0.8570,
        )
        self.tracker.log_run(
            model_name="M1",
            seed=42,
            started_at=start_time,
            ended_at=end_time,
            epochs_trained=10,
            best_epoch=6,
            trainable_params=300000,
            total_params=300000,
            hyperparameters={"lr": 0.001, "batch_size": 64, "bidirectional": True},
            dataset_content_sha256="dummy_hash_123",
            split_indices_sha256={"train": "hash_tr", "val": "hash_v", "test": "hash_te"},
            metrics=metrics_m1_s42,
        )

        table = self.tracker.generate_comparison_table()
        self.assertEqual(len(table), 2)
        self.assertEqual(table.iloc[0]["Model"], "M0")
        self.assertEqual(table.iloc[1]["Model"], "M1")
        self.assertEqual(table.iloc[1]["Δ vs Previous"], "+0.0160")
        self.assertEqual(table.iloc[1]["Δ vs M0"], "+0.0160")


if __name__ == "__main__":
    unittest.main()
