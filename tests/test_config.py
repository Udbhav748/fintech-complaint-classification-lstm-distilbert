"""Unit tests for configuration loading and validation."""

import unittest

from src.checkpoint import VALID_EXPERIMENTS
from src.config import (
    ConfigValidationError,
    get_experiment_config,
    load_data_config,
    validate_experiment_config,
)


class TestConfig(unittest.TestCase):
    def test_load_data_config(self):
        cfg = load_data_config()
        self.assertIn("dataset", cfg)
        self.assertIn("cleaning", cfg)
        self.assertIn("split", cfg)
        self.assertEqual(cfg["dataset"]["rows_raw"], 107992)
        self.assertEqual(cfg["dataset"]["rows_after_dedup"], 101802)

    def test_get_and_validate_all_experiment_configs(self):
        for exp in VALID_EXPERIMENTS:
            cfg = get_experiment_config(exp)
            self.assertEqual(cfg["experiment"], exp)
            validate_experiment_config(cfg)

    def test_rejects_invalid_experiment_or_missing_keys(self):
        with self.assertRaises(ConfigValidationError):
            get_experiment_config("custom_alias_v1")

        bad_cfg = {"experiment": "M0", "model_type": "lstm"}  # missing batch_size etc.
        with self.assertRaises(ConfigValidationError):
            validate_experiment_config(bad_cfg)


if __name__ == "__main__":
    unittest.main()
