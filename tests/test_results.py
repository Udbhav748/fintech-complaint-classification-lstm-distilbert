"""Unit tests for results infrastructure, schema validation, and comparison table generation."""

from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from src.checkpoint import save_checkpoint_record
from src.fingerprint import SplitFingerprint
from src.results import (
    RUNS_SCHEMA,
    ResultsValidationError,
    generate_comparison_table,
    log_run_to_csv,
    validate_runs_csv,
    verify_run_traceability,
)


class TestResultsInfrastructure(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.runs_csv = self.temp_dir / "runs.csv"

        self.sample_run_m0 = {
            "experiment": "M0",
            "model": "LSTM baseline",
            "seed": 42,
            "macro_f1": 0.8420,
            "accuracy": 0.8450,
            "macro_precision": 0.8430,
            "macro_recall": 0.8410,
            "best_epoch": 7,
            "epochs_run": 10,
            "train_time": 125.4,
            "parameter_count": 150000,
            "checkpoint_path": "checkpoints/M0_seed42.weights.h5",
            "dataset_version": "149a4b98ba4d4bbcb1f15ca74da93bf4529be9755a22b19603d85f93e69df6a4",
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_run_and_validate_success(self):
        log_run_to_csv(self.sample_run_m0, runs_csv_path=self.runs_csv)
        self.assertTrue(self.runs_csv.exists())

        is_valid, errors = validate_runs_csv(self.runs_csv)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_rejects_invalid_experiment_name_or_alias(self):
        bad_run = self.sample_run_m0.copy()
        bad_run["experiment"] = "final_model_v2"  # illegal alias

        with self.assertRaises(ResultsValidationError) as ctx:
            log_run_to_csv(bad_run, runs_csv_path=self.runs_csv)
        self.assertIn("Invalid experiment names detected", str(ctx.exception))

    def test_rejects_out_of_range_metrics(self):
        bad_run = self.sample_run_m0.copy()
        bad_run["macro_f1"] = 1.05  # > 1.0

        with self.assertRaises(ResultsValidationError) as ctx:
            log_run_to_csv(bad_run, runs_csv_path=self.runs_csv)
        self.assertIn("Metric 'macro_f1' out of valid [0, 1] range", str(ctx.exception))

    def test_rejects_best_epoch_greater_than_epochs_run(self):
        bad_run = self.sample_run_m0.copy()
        bad_run["best_epoch"] = 12
        bad_run["epochs_run"] = 10

        with self.assertRaises(ResultsValidationError) as ctx:
            log_run_to_csv(bad_run, runs_csv_path=self.runs_csv)
        self.assertIn("best_epoch > epochs_run", str(ctx.exception))

    def test_generate_comparison_table_with_multi_seed_and_deltas(self):
        # Log 3 seeds for M0 (stability reference)
        for seed, f1 in [(42, 0.8410), (123, 0.8430), (456, 0.8420)]:
            run = self.sample_run_m0.copy()
            run["seed"] = seed
            run["macro_f1"] = f1
            run["accuracy"] = f1 + 0.0030
            log_run_to_csv(run, runs_csv_path=self.runs_csv)

        # Log 1 seed for M1
        run_m1 = self.sample_run_m0.copy()
        run_m1["experiment"] = "M1"
        run_m1["model"] = "Bidirectional LSTM"
        run_m1["seed"] = 42
        run_m1["macro_f1"] = 0.8580
        run_m1["accuracy"] = 0.8610
        log_run_to_csv(run_m1, runs_csv_path=self.runs_csv)

        table = generate_comparison_table(runs_csv_path=self.runs_csv)
        self.assertEqual(len(table), 2)

        # Row 0: M0
        m0_row = table.iloc[0]
        self.assertEqual(m0_row["Model"], "M0")
        self.assertIn("±", m0_row["Macro-F1"])
        self.assertEqual(m0_row["Δ vs Previous"], "—")
        self.assertEqual(m0_row["Δ vs M0"], "—")

        # Row 1: M1
        m1_row = table.iloc[1]
        self.assertEqual(m1_row["Model"], "M1")
        self.assertEqual(m1_row["Macro-F1"], "0.8580")
        # M0 mean = 0.8420, M1 = 0.8580 -> delta = +0.0160
        self.assertEqual(m1_row["Δ vs Previous"], "+0.0160")
        self.assertEqual(m1_row["Δ vs M0"], "+0.0160")


class TestRunTraceability(unittest.TestCase):
    """Part 9: a result row must trace to the dataset it was actually run on and
    to a checkpoint that was actually selected with val_macro_f1."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.split_manifest = self.temp_dir / "split_manifest.json"

        self.dataset_hash = "a" * 64
        SplitFingerprint(
            dataset_content_sha256=self.dataset_hash,
            train_indices_sha256="b" * 64,
            val_indices_sha256="c" * 64,
            test_indices_sha256="d" * 64,
            train_count=8, val_count=1, test_count=1, total_count=10,
            train_class_distribution={}, val_class_distribution={}, test_class_distribution={},
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            disjoint_indices_verified=True, zero_duplicate_cross_split_verified=True,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        ).save(self.split_manifest)

        ckpt_file = self.checkpoint_dir / "M0_seed42.weights.h5"
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        ckpt_file.write_bytes(b"dummy")
        save_checkpoint_record(
            experiment="M0", seed=42, best_epoch=3, best_val_macro_f1=0.84,
            checkpoint_file=ckpt_file, checkpoint_dir=self.checkpoint_dir,
        )

        self.run = {
            "experiment": "M0", "seed": 42, "dataset_version": self.dataset_hash,
            "checkpoint_path": str(ckpt_file),
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_traceable_run_passes(self):
        ok, errors = verify_run_traceability(self.run, self.split_manifest, self.checkpoint_dir)
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_wrong_dataset_version_fails(self):
        bad_run = {**self.run, "dataset_version": "f" * 64}
        ok, errors = verify_run_traceability(bad_run, self.split_manifest, self.checkpoint_dir)
        self.assertFalse(ok)
        self.assertTrue(any("dataset_content_sha256" in e for e in errors))

    def test_missing_checkpoint_metadata_fails(self):
        bad_run = {**self.run, "seed": 999}
        ok, errors = verify_run_traceability(bad_run, self.split_manifest, self.checkpoint_dir)
        self.assertFalse(ok)
        self.assertTrue(any("No checkpoint metadata found" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
