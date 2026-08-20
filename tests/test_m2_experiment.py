"""M2-specific integration checks (project_plan.md Stage 8).

Generic config/schema/delta/traceability behavior is already covered by
tests/test_config.py, tests/test_results.py, and tests/test_models.py (which
already builds M2 via the factory and checks SpatialDropout1D + LSTM
dropout/recurrent_dropout). This file checks the one thing those don't: that
M2's registered configuration is *exactly* M1 plus the two locked regularization
values - no other field silently drifted - and that the registered result's
deltas match the locked M0/M1 values.
"""

import unittest

from src.config import EXPERIMENT_CONFIGS, get_experiment_config
from src.evaluation import calculate_deltas

try:
    from src.models import RecurrentModelConfig, build_recurrent_model, count_parameters

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TestM2IsOnlyRegularizationChange(unittest.TestCase):
    """project_plan.md §14.10: 'The only intended change from M1 is the
    addition of the planned regularization.'"""

    # M1 and M2 both use seed 42, so "seeds" happens not to differ here - it is
    # still allowed to (single-seed rungs aren't required to share a seed), just
    # not required to. Everything else in this set must actually change.
    ALLOWED_DIFFERENCES = {"experiment", "model_type", "description", "seeds"}
    REQUIRED_DIFFERENCES = {"dropout", "recurrent_dropout", "spatial_dropout"}

    def test_only_regularization_and_bookkeeping_fields_differ(self):
        m1 = get_experiment_config("M1")
        m2 = get_experiment_config("M2")
        all_keys = set(m1) | set(m2)
        differing = {k for k in all_keys if m1.get(k) != m2.get(k)}
        unexpected = differing - self.ALLOWED_DIFFERENCES - self.REQUIRED_DIFFERENCES
        self.assertEqual(unexpected, set(), f"M1->M2 changed unexpected fields: {unexpected}")
        self.assertTrue(
            self.REQUIRED_DIFFERENCES.issubset(differing),
            f"M1->M2 must change {self.REQUIRED_DIFFERENCES}, only changed {differing}",
        )

    def test_m2_dropout_and_recurrent_dropout_locked_values(self):
        cfg = EXPERIMENT_CONFIGS["M2"]
        self.assertEqual(cfg["dropout"], 0.3)
        self.assertEqual(cfg["recurrent_dropout"], 0.2)
        self.assertEqual(cfg["spatial_dropout"], 0.2)

    def test_m2_still_bidirectional_random_embedding_max_len_128(self):
        cfg = EXPERIMENT_CONFIGS["M2"]
        self.assertTrue(cfg["bidirectional"])
        self.assertFalse(cfg["use_glove"])
        self.assertEqual(cfg["max_len"], 128)

    def test_m2_seed_policy(self):
        self.assertEqual(EXPERIMENT_CONFIGS["M2"]["seeds"], [42])  # single-seed rung

    def test_m2_no_early_stopping_or_lr_schedule(self):
        cfg = EXPERIMENT_CONFIGS["M2"]
        self.assertFalse(cfg["early_stopping"])
        self.assertNotIn("lr_schedule", cfg)


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestM2ArchitectureMatchesRealConfig(unittest.TestCase):
    """Built from the real (non-tiny) M1/M2 configs, not synthetic dimensions -
    confirms parameter count is unchanged (regularization adds no weights)."""

    def test_parameter_count_identical_to_m1(self):
        m1 = build_recurrent_model(RecurrentModelConfig.from_experiment("M1"))
        m2 = build_recurrent_model(RecurrentModelConfig.from_experiment("M2"))
        self.assertEqual(count_parameters(m1)["total"], count_parameters(m2)["total"])

    def test_m2_has_spatial_dropout_layer_with_correct_rate(self):
        m2 = build_recurrent_model(RecurrentModelConfig.from_experiment("M2"))
        spatial = m2.get_layer("spatial_dropout")
        self.assertAlmostEqual(spatial.rate, 0.2)

    def test_m2_lstm_dropout_and_recurrent_dropout(self):
        m2 = build_recurrent_model(RecurrentModelConfig.from_experiment("M2"))
        bidir = m2.get_layer("bidirectional_lstm")
        self.assertEqual(bidir.__class__.__name__, "Bidirectional")
        self.assertEqual(bidir.forward_layer.__class__.__name__, "LSTM")
        self.assertAlmostEqual(bidir.forward_layer.dropout, 0.3)
        self.assertAlmostEqual(bidir.forward_layer.recurrent_dropout, 0.2)

    def test_m2_input_and_output_shape(self):
        m2 = build_recurrent_model(RecurrentModelConfig.from_experiment("M2"))
        self.assertEqual(m2.input_shape, (None, 128))
        self.assertEqual(m2.output_shape, (None, 5))


class TestM2DeltaCalculation(unittest.TestCase):
    """The registered M2 result (project_plan.md §14.10):
    macro_f1=0.8517974977710514 against M1=0.8485154570891954 and
    M0 mean=0.8508."""

    def test_m2_minus_m1_delta_matches_registered_value(self):
        m2_macro_f1 = 0.8517974977710514
        m1_macro_f1 = 0.8485154570891954
        deltas = calculate_deltas(current_f1=m2_macro_f1, baseline_m0_f1=m1_macro_f1)
        self.assertAlmostEqual(deltas["delta_vs_m0"], 0.003282040681856002, places=12)

    def test_m2_minus_m0_delta_matches_registered_value(self):
        m2_macro_f1 = 0.8517974977710514
        m0_mean = 0.8508
        deltas = calculate_deltas(current_f1=m2_macro_f1, baseline_m0_f1=m0_mean)
        self.assertAlmostEqual(deltas["delta_vs_m0"], 0.0009974977710514032, places=12)

    def test_m2_minus_m1_delta_is_smaller_than_m0_stability_spread(self):
        # project_plan.md §9: M0's own three-seed min-max spread is 0.0041.
        m2_m1_delta = abs(0.003282040681856002)
        m0_seed_spread = 0.0041
        self.assertLess(m2_m1_delta, m0_seed_spread)


if __name__ == "__main__":
    unittest.main()
