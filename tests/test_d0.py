"""D0-specific integration checks (project_plan.md Stage 12).

Generic schema/delta/traceability behavior is already covered by
tests/test_results.py. This file checks what's specific to D0: the config
matches the locked spec, the model/tokenizer build correctly and are
compatible with each other, `ValMacroF1Callback` computes the same value
`compute_val_macro_f1` would, and D0's config work never touched M0-M4.
"""

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.checkpoint import VALID_EXPERIMENTS
from src.config import EXPERIMENT_CONFIGS, get_experiment_config
from src.data import CLASS_TO_ID, LABELS
from src.evaluation import compute_val_macro_f1
from src.results import RUNS_SCHEMA, validate_runs_csv

try:
    import tf_keras
    import transformers  # noqa: F401

    from src.distilbert import (
        ValMacroF1Callback, build_d0_callbacks, build_d0_model, build_d0_tokenizer,
        compile_d0_model, verify_d0_config,
    )

    D0_STACK_AVAILABLE = True
except ImportError:
    D0_STACK_AVAILABLE = False


class TestD0Config(unittest.TestCase):
    """No TF/transformers needed - pure config checks."""

    def test_checkpoint_and_max_len_locked(self):
        cfg = EXPERIMENT_CONFIGS["D0"]
        self.assertEqual(cfg["pretrained_model_name"], "distilbert-base-uncased")
        self.assertEqual(cfg["max_len"], 256)

    def test_three_seed_policy(self):
        self.assertEqual(EXPERIMENT_CONFIGS["D0"]["seeds"], [42, 123, 456])

    def test_training_hyperparameters_present(self):
        cfg = EXPERIMENT_CONFIGS["D0"]
        for key in ("optimizer", "learning_rate", "batch_size", "epochs",
                    "weight_decay", "early_stopping", "early_stopping_patience"):
            self.assertIn(key, cfg)
        self.assertEqual(cfg["optimizer"], "adamw")
        self.assertEqual(cfg["epochs"], 4)
        self.assertTrue(cfg["early_stopping"])
        self.assertEqual(cfg["early_stopping_patience"], 2)

    def test_d0_is_valid_experiment_id(self):
        self.assertIn("D0", VALID_EXPERIMENTS)

    def test_d0_work_did_not_touch_recurrent_experiment_definitions(self):
        # Spot-check the locked M0-M4 values this stage must not have edited.
        self.assertEqual(EXPERIMENT_CONFIGS["M0"]["bidirectional"], False)
        self.assertEqual(EXPERIMENT_CONFIGS["M1"]["bidirectional"], True)
        self.assertEqual(EXPERIMENT_CONFIGS["M2"]["dropout"], 0.3)
        self.assertEqual(EXPERIMENT_CONFIGS["M2"]["recurrent_dropout"], 0.2)
        self.assertTrue(EXPERIMENT_CONFIGS["M3"]["use_glove"])
        self.assertEqual(EXPERIMENT_CONFIGS["M4"]["max_len"], 256)
        self.assertEqual(EXPERIMENT_CONFIGS["M4"]["epochs"], 10)  # the Task 11 bug fix must still hold
        for exp in ["M0", "M1", "M2", "M3", "M4"]:
            self.assertEqual(EXPERIMENT_CONFIGS[exp]["seeds"], get_experiment_config(exp)["seeds"])


@unittest.skipUnless(D0_STACK_AVAILABLE, "tensorflow/tf_keras/transformers not installed in this environment")
class TestD0ModelTokenizerCompatibility(unittest.TestCase):
    """Part 2: built once per class (expensive - a real 67M-param download/
    load), reused across assertions rather than rebuilt per test."""

    @classmethod
    def setUpClass(cls):
        cfg = get_experiment_config("D0")
        cls.checkpoint, cls.max_len = cfg["pretrained_model_name"], cfg["max_len"]
        cls.tokenizer = build_d0_tokenizer(cls.checkpoint)
        cls.model = build_d0_model(cls.checkpoint, num_labels=len(LABELS))

    def test_verify_d0_config_passes_for_the_real_locked_values(self):
        verify_d0_config(self.checkpoint, self.max_len, num_labels=len(LABELS))

    def test_verify_d0_config_rejects_wrong_checkpoint(self):
        with self.assertRaises(ValueError):
            verify_d0_config("bert-base-uncased", self.max_len, num_labels=len(LABELS))

    def test_verify_d0_config_rejects_wrong_num_labels(self):
        with self.assertRaises(ValueError):
            verify_d0_config(self.checkpoint, self.max_len, num_labels=3)

    def test_tokenizer_has_required_special_tokens(self):
        for tok in ("cls_token", "sep_token", "pad_token"):
            self.assertIn(tok, self.tokenizer.special_tokens_map)

    def test_input_ids_and_attention_mask_shape(self):
        encoded = self.tokenizer(["a", "b", "c"], truncation=True, max_length=self.max_len,
                                  padding="max_length", return_tensors="np")
        self.assertEqual(encoded["input_ids"].shape, (3, self.max_len))
        self.assertEqual(encoded["attention_mask"].shape, (3, self.max_len))

    def test_model_output_classes(self):
        self.assertEqual(self.model.config.num_labels, len(LABELS))
        enc = self.tokenizer(["test"], truncation=True, max_length=self.max_len,
                              padding="max_length", return_tensors="np")
        out = self.model({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        self.assertEqual(out.logits.shape, (1, len(LABELS)))

    def test_canonical_label_mapping_used_by_class_to_id(self):
        self.assertEqual(len(CLASS_TO_ID), len(LABELS))
        self.assertEqual(set(CLASS_TO_ID.values()), set(range(len(LABELS))))


@unittest.skipUnless(D0_STACK_AVAILABLE, "tensorflow/tf_keras/transformers not installed in this environment")
class TestD0CheckpointRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_and_load_weights_round_trip(self):
        cfg = get_experiment_config("D0")
        model = build_d0_model(cfg["pretrained_model_name"], num_labels=len(LABELS))
        model = compile_d0_model(model, learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        ckpt_path = self.tmp / "D0_seed42.weights.h5"
        model.save_weights(str(ckpt_path))
        self.assertTrue(ckpt_path.exists())
        self.assertGreater(ckpt_path.stat().st_size, 0)

        reloaded = build_d0_model(cfg["pretrained_model_name"], num_labels=len(LABELS))
        reloaded = compile_d0_model(reloaded, learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        reloaded.load_weights(str(ckpt_path))
        for w1, w2 in zip(model.weights, reloaded.weights):
            np.testing.assert_allclose(w1.numpy(), w2.numpy())


@unittest.skipUnless(D0_STACK_AVAILABLE, "tensorflow/tf_keras/transformers not installed in this environment")
class TestValMacroF1Callback(unittest.TestCase):
    """Confirms the callback writes the same value compute_val_macro_f1 would
    compute directly - not a second metric implementation, just checked at
    the callback's own call site."""

    def test_on_epoch_end_matches_direct_compute_val_macro_f1(self):
        class FakeOutput:
            def __init__(self, logits):
                self.logits = logits

        class FakeModel:
            def predict(self, x, verbose=0):
                # Deterministic "predictions": argmax always matches y_true for
                # indices 0-3, wrong for index 4 - gives a non-trivial macro-F1.
                logits = np.eye(5, dtype=np.float32)[np.array([0, 1, 2, 3, 0])]
                return FakeOutput(logits)

        val_labels = np.array([0, 1, 2, 3, 4])
        callback = ValMacroF1Callback(
            val_input_ids=np.zeros((5, 10)), val_attention_mask=np.ones((5, 10)), val_labels=val_labels
        )
        callback.model = FakeModel()

        logs = {}
        callback.on_epoch_end(epoch=0, logs=logs)

        expected_pred = np.array([0, 1, 2, 3, 0])
        expected = compute_val_macro_f1(val_labels, expected_pred, labels=LABELS)
        self.assertIn("val_macro_f1", logs)
        self.assertAlmostEqual(logs["val_macro_f1"], expected, places=10)


class TestD0ResultSchemaAndTraceability(unittest.TestCase):
    """Synthetic run dict, same pattern as tests/test_results.py's M0 example
    - schema validation only, no model involved."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.runs_csv = self.temp_dir / "runs.csv"
        self.sample_d0_run = {
            "experiment": "D0", "model": "DistilBERT (fine-tuned transformer)", "seed": 42,
            "macro_f1": 0.88, "accuracy": 0.87, "macro_precision": 0.88, "macro_recall": 0.88,
            "best_epoch": 2, "epochs_run": 4, "train_time": 3600.0, "parameter_count": 66957317,
            "checkpoint_path": "checkpoints/D0_seed42.weights.h5",
            "dataset_version": "149a4b98ba4d4bbcb1f15ca74da93bf4529be9755a22b19603d85f93e69df6a4",
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_d0_row_matches_runs_schema(self):
        for field in RUNS_SCHEMA:
            self.assertIn(field, self.sample_d0_run)

    def test_d0_row_validates_via_log_run_to_csv(self):
        from src.results import log_run_to_csv

        log_run_to_csv(self.sample_d0_run, runs_csv_path=self.runs_csv)
        is_valid, errors = validate_runs_csv(self.runs_csv)
        self.assertTrue(is_valid, errors)

    def test_three_d0_seeds_satisfy_seed_completeness(self):
        from src.results import log_run_to_csv

        for seed in [42, 123, 456]:
            run = {**self.sample_d0_run, "seed": seed}
            log_run_to_csv(run, runs_csv_path=self.runs_csv)
        df = pd.read_csv(self.runs_csv)
        self.assertEqual(len(df[df["experiment"] == "D0"]), 3)


RUNS_CSV_PATH = Path("results/runs.csv")


@unittest.skipUnless(RUNS_CSV_PATH.exists(), "results/runs.csv not present in this checkout")
class TestD0RegisteredResults(unittest.TestCase):
    """project_plan.md §14.14: the actual registered 3-seed D0 result. Checks
    the committed values, not a synthetic example - catches accidental edits
    to results/runs.csv for D0's rows."""

    def setUp(self):
        self.df = pd.read_csv(RUNS_CSV_PATH)
        self.d0 = self.df[self.df["experiment"] == "D0"]

    def test_d0_has_exactly_three_seeds(self):
        self.assertEqual(sorted(self.d0["seed"].tolist()), [42, 123, 456])

    def test_d0_every_seed_exceeds_m4(self):
        m4_f1 = self.df[self.df["experiment"] == "M4"]["macro_f1"].iloc[0]
        for _, row in self.d0.iterrows():
            self.assertGreater(row["macro_f1"], m4_f1)

    def test_d0_mean_and_std_match_independent_recompute(self):
        from src.evaluation import seed_statistics

        stats = seed_statistics(self.d0["macro_f1"].values)
        self.assertAlmostEqual(stats["mean"], 0.8758532501836456, places=10)
        self.assertAlmostEqual(stats["std"], 0.0018327384663822041, places=10)

    def test_d0_minus_m4_delta(self):
        from src.evaluation import calculate_deltas, seed_statistics

        m4_f1 = self.df[self.df["experiment"] == "M4"]["macro_f1"].iloc[0]
        d0_mean = seed_statistics(self.d0["macro_f1"].values)["mean"]
        delta = calculate_deltas(current_f1=d0_mean, baseline_m0_f1=m4_f1)["delta_vs_m0"]
        self.assertAlmostEqual(delta, 0.00482950383535119, places=10)

    def test_d0_minus_m0_delta(self):
        from src.evaluation import calculate_deltas, seed_statistics

        m0_mean = seed_statistics(self.df[self.df["experiment"] == "M0"]["macro_f1"].values)["mean"]
        d0_mean = seed_statistics(self.d0["macro_f1"].values)["mean"]
        delta = calculate_deltas(current_f1=d0_mean, baseline_m0_f1=m0_mean)["delta_vs_m0"]
        self.assertAlmostEqual(delta, 0.025020171409647518, places=10)

    def test_d0_parameter_count_identical_across_seeds(self):
        self.assertEqual(self.d0["parameter_count"].nunique(), 1)
        self.assertEqual(self.d0["parameter_count"].iloc[0], 66957317)

    def test_m0_to_m4_rows_untouched_by_d0_work(self):
        # Same values every prior task's own test file asserts - a change here
        # would mean D0 work accidentally modified a prior result.
        m4 = self.df[self.df["experiment"] == "M4"].iloc[0]
        self.assertAlmostEqual(m4["macro_f1"], 0.8710237463482944, places=12)


if __name__ == "__main__":
    unittest.main()
