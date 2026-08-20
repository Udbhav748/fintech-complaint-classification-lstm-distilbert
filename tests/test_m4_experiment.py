"""M4-specific integration checks (project_plan.md Stage 10).

Generic config/schema/delta/traceability/callback behavior is already covered
by tests/test_config.py, tests/test_results.py, and tests/test_models.py
(which already builds M4 via the factory, checks the recurrent_dropout/
lr_schedule locked values, and checks the M4 callback list order). This file
checks the one thing those don't: that M4's registered configuration is
*exactly* M3 plus the three bundled changes - no other field silently
drifted - that the epoch budget stays at the project-wide 10-epoch ceiling
(the bug found and fixed during this stage's pre-flight), and that the
registered result's deltas match the locked M0/M3 values.
"""

import unittest

from src.config import EXPERIMENT_CONFIGS, get_experiment_config
from src.evaluation import calculate_deltas

try:
    from src.models import RecurrentModelConfig, build_callbacks, build_recurrent_model, count_parameters

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TestM4IsOnlyBundledOptimizationContextChange(unittest.TestCase):
    """project_plan.md §14.12: 'M4 adds exactly three bundled changes:
    max_len=256, ReduceLROnPlateau, EarlyStopping.'"""

    ALLOWED_DIFFERENCES = {"experiment", "model_type", "description", "seeds"}
    REQUIRED_DIFFERENCES = {
        "max_len", "lr_schedule", "lr_schedule_factor", "lr_schedule_patience",
        "lr_schedule_min_lr", "early_stopping", "early_stopping_patience",
        "early_stopping_restore_best_weights",
    }

    def test_only_bundled_stage_and_bookkeeping_fields_differ(self):
        m3 = get_experiment_config("M3")
        m4 = get_experiment_config("M4")
        all_keys = set(m3) | set(m4)
        differing = {k for k in all_keys if m3.get(k) != m4.get(k)}
        unexpected = differing - self.ALLOWED_DIFFERENCES - self.REQUIRED_DIFFERENCES
        self.assertEqual(unexpected, set(), f"M3->M4 changed unexpected fields: {unexpected}")
        self.assertTrue(
            self.REQUIRED_DIFFERENCES.issubset(differing),
            f"M3->M4 must change {self.REQUIRED_DIFFERENCES}, only changed {differing}",
        )

    def test_m4_keeps_m3_regularization_and_embedding_unchanged(self):
        cfg = EXPERIMENT_CONFIGS["M4"]
        self.assertEqual(cfg["dropout"], 0.3)
        self.assertEqual(cfg["recurrent_dropout"], 0.2)
        self.assertEqual(cfg["spatial_dropout"], 0.2)
        self.assertTrue(cfg["use_glove"])
        self.assertTrue(cfg["bidirectional"])

    def test_m4_max_len_256(self):
        self.assertEqual(EXPERIMENT_CONFIGS["M4"]["max_len"], 256)

    def test_m4_scheduler_locked_values(self):
        cfg = EXPERIMENT_CONFIGS["M4"]
        self.assertEqual(cfg["lr_schedule"], "reduce_on_plateau")
        self.assertEqual(cfg["lr_schedule_factor"], 0.5)
        self.assertEqual(cfg["lr_schedule_patience"], 2)
        self.assertEqual(cfg["lr_schedule_min_lr"], 1e-5)

    def test_m4_early_stopping_locked_values(self):
        cfg = EXPERIMENT_CONFIGS["M4"]
        self.assertTrue(cfg["early_stopping"])
        self.assertEqual(cfg["early_stopping_patience"], 3)
        self.assertTrue(cfg["early_stopping_restore_best_weights"])

    def test_m4_seed_policy(self):
        self.assertEqual(EXPERIMENT_CONFIGS["M4"]["seeds"], [42])  # single-seed rung

    def test_m4_epoch_budget_matches_project_wide_ceiling(self):
        # The bug this stage found and fixed pre-flight: this was 20, which
        # contradicted project_plan.md §4/§13's "max 10, ladder-wide" rule.
        self.assertEqual(EXPERIMENT_CONFIGS["M4"]["epochs"], 10)
        for exp in ["M0", "M1", "M2", "M3", "M4"]:
            self.assertEqual(
                EXPERIMENT_CONFIGS[exp]["epochs"], 10,
                f"{exp} epoch budget must be 10, ladder-wide - not per-model",
            )


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestM4ArchitectureMatchesRealConfig(unittest.TestCase):
    """Built from the real (non-tiny) M3/M4 configs and the real frozen
    GloVe matrix, not synthetic dimensions."""

    def _load_matrix(self):
        import numpy as np

        return np.load("artifacts/embeddings/glove_100d_matrix.npy")

    def test_parameter_count_identical_to_m3(self):
        matrix = self._load_matrix()
        m3 = build_recurrent_model(RecurrentModelConfig.from_experiment("M3"), embedding_matrix=matrix)
        m4 = build_recurrent_model(RecurrentModelConfig.from_experiment("M4"), embedding_matrix=matrix)
        self.assertEqual(count_parameters(m3)["total"], count_parameters(m4)["total"])

    def test_m4_input_shape_is_256(self):
        m4 = build_recurrent_model(RecurrentModelConfig.from_experiment("M4"), embedding_matrix=self._load_matrix())
        self.assertEqual(m4.input_shape, (None, 256))
        self.assertEqual(m4.output_shape, (None, 5))

    def test_m4_no_stacked_recurrent_layers(self):
        m4 = build_recurrent_model(RecurrentModelConfig.from_experiment("M4"), embedding_matrix=self._load_matrix())
        lstm_like = [l for l in m4.layers if "lstm" in l.name.lower()]
        self.assertEqual(len(lstm_like), 1, "M4 must not have stacked recurrent layers")

    def test_m4_callback_order_and_monitor(self):
        cfg = RecurrentModelConfig.from_experiment("M4")
        callbacks = build_callbacks(cfg, "M4_seed42.weights.h5")
        self.assertEqual(
            [c.__class__.__name__ for c in callbacks],
            ["ModelCheckpoint", "ReduceLROnPlateau", "EarlyStopping"],
        )
        for c in callbacks:
            self.assertEqual(c.monitor, "val_macro_f1")
        self.assertEqual(callbacks[1].factor, 0.5)
        self.assertEqual(callbacks[1].patience, 2)
        self.assertEqual(callbacks[1].min_lr, 1e-5)
        self.assertEqual(callbacks[2].patience, 3)
        self.assertTrue(callbacks[2].restore_best_weights)


class TestM4DeltaCalculation(unittest.TestCase):
    """The registered M4 result (project_plan.md §14.12):
    macro_f1=0.8710237463482944 against M3=0.8656677597625515 and
    M0 mean=0.8508."""

    def test_m4_minus_m3_delta_matches_registered_value(self):
        m4_macro_f1 = 0.8710237463482944
        m3_macro_f1 = 0.8656677597625515
        deltas = calculate_deltas(current_f1=m4_macro_f1, baseline_m0_f1=m3_macro_f1)
        self.assertAlmostEqual(deltas["delta_vs_m0"], 0.005355986585742878, places=12)

    def test_m4_minus_m0_delta_matches_registered_value(self):
        m4_macro_f1 = 0.8710237463482944
        m0_mean = 0.8508
        deltas = calculate_deltas(current_f1=m4_macro_f1, baseline_m0_f1=m0_mean)
        self.assertAlmostEqual(deltas["delta_vs_m0"], 0.02022374634829438, places=12)

    def test_m4_minus_m3_delta_exceeds_m0_stability_spread(self):
        # project_plan.md §9: M0's own three-seed min-max spread is 0.0041.
        m4_m3_delta = abs(0.005355986585742878)
        m0_seed_spread = 0.0041
        self.assertGreater(m4_m3_delta, m0_seed_spread)


if __name__ == "__main__":
    unittest.main()
