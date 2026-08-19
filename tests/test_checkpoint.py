"""Unit tests for checkpoint management, path determinism, and validation."""

from pathlib import Path
import shutil
import tempfile
import unittest

from src.checkpoint import (
    CheckpointMetadata,
    CheckpointValidationError,
    get_checkpoint_path,
    get_metadata_path,
    save_checkpoint_record,
    verify_checkpoint,
)


class TestCheckpointValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deterministic_checkpoint_paths(self):
        p1 = get_checkpoint_path("M0", seed=42, checkpoint_dir=self.temp_dir)
        p2 = get_checkpoint_path("M0", seed=42, checkpoint_dir=self.temp_dir)
        self.assertEqual(p1, p2)
        self.assertEqual(p1.name, "M0_seed42.weights.h5")

        with self.assertRaises(ValueError):
            get_checkpoint_path("invalid_model_v2", seed=42, checkpoint_dir=self.temp_dir)

    def test_save_and_verify_checkpoint_success(self):
        ckpt_file = get_checkpoint_path("M1", seed=123, checkpoint_dir=self.temp_dir)
        # Create non-empty weights file
        ckpt_file.write_bytes(b"dummy_model_weights_bytes_123456")

        meta_path = save_checkpoint_record(
            experiment="M1",
            seed=123,
            best_epoch=5,
            best_val_macro_f1=0.8580,
            checkpoint_file=ckpt_file,
            model_architecture={"hidden_dim": 128, "bidirectional": True},
            checkpoint_dir=self.temp_dir,
            monitor_metric="val_macro_f1",
        )
        self.assertTrue(meta_path.exists())

        is_valid, errors = verify_checkpoint("M1", seed=123, checkpoint_dir=self.temp_dir)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_rejects_non_val_macro_f1_metric(self):
        ckpt_file = get_checkpoint_path("M0", seed=42, checkpoint_dir=self.temp_dir)
        ckpt_file.write_bytes(b"dummy_weights")

        with self.assertRaises(CheckpointValidationError):
            save_checkpoint_record(
                experiment="M0",
                seed=42,
                best_epoch=1,
                best_val_macro_f1=0.80,
                checkpoint_file=ckpt_file,
                checkpoint_dir=self.temp_dir,
                monitor_metric="val_loss",  # Policy requires val_macro_f1
            )

    def test_fails_on_missing_weights_file(self):
        # Save metadata but don't create weights file
        meta = CheckpointMetadata(
            experiment="M2",
            seed=42,
            best_epoch=3,
            monitor_metric="val_macro_f1",
            best_metric_value=0.8550,
            checkpoint_file="non_existent.weights.h5",
            model_architecture={},
            created_at_utc="2026-08-20T00:00:00Z",
        )
        meta.save(get_metadata_path("M2", seed=42, checkpoint_dir=self.temp_dir))

        with self.assertRaises(CheckpointValidationError) as ctx:
            verify_checkpoint("M2", seed=42, checkpoint_dir=self.temp_dir)
        self.assertIn("Checkpoint file not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
