"""M1-specific integration checks (project_plan.md Stage 7).

Generic config/schema/delta/traceability behavior is already covered by
tests/test_config.py, tests/test_results.py, and tests/test_models.py (which
already builds M1 via the factory and checks the Bidirectional wrapper). This
file checks the one thing those don't: that M1's registered configuration is
*exactly* M0 plus the single Bidirectional change - no other field silently
drifted - and that the registered result's delta matches the locked M0 mean.
"""

import unittest

from src.config import EXPERIMENT_CONFIGS, get_experiment_config
from src.evaluation import calculate_deltas

try:
    from src.models import RecurrentModelConfig, build_recurrent_model, count_parameters

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TestM1IsOnlyBidirectionalChange(unittest.TestCase):
    """project_plan.md §14.9: 'The only intended model change from M0 is
    LSTM -> Bidirectional(LSTM). Everything else must remain fixed.'"""

    # Fields allowed to differ between M0 and M1: bidirectional itself, and the
    # bookkeeping fields (identity/description/seed count) that were never part
    # of the architecture in the first place.
    ALLOWED_DIFFERENCES = {"experiment", "model_type", "description", "bidirectional", "seeds"}

    def test_only_bidirectional_and_bookkeeping_fields_differ(self):
        m0 = get_experiment_config("M0")
        m1 = get_experiment_config("M1")
        all_keys = set(m0) | set(m1)
        differing = {k for k in all_keys if m0.get(k) != m1.get(k)}
        self.assertEqual(
            differing, self.ALLOWED_DIFFERENCES,
            f"M0->M1 changed unexpected fields: {differing - self.ALLOWED_DIFFERENCES}",
        )

    def test_m1_bidirectional_is_true_m0_is_false(self):
        self.assertFalse(EXPERIMENT_CONFIGS["M0"]["bidirectional"])
        self.assertTrue(EXPERIMENT_CONFIGS["M1"]["bidirectional"])

    def test_m1_has_no_dropout_no_recurrent_dropout_no_glove(self):
        cfg = EXPERIMENT_CONFIGS["M1"]
        self.assertEqual(cfg["dropout"], 0.0)
        self.assertEqual(cfg.get("recurrent_dropout", 0.0), 0.0)
        self.assertEqual(cfg["spatial_dropout"], 0.0)
        self.assertFalse(cfg["use_glove"])

    def test_m1_max_len_and_seed_policy(self):
        cfg = EXPERIMENT_CONFIGS["M1"]
        self.assertEqual(cfg["max_len"], 128)
        self.assertEqual(cfg["seeds"], [42])  # single-seed rung, not invented

    def test_m1_no_early_stopping_or_lr_schedule(self):
        cfg = EXPERIMENT_CONFIGS["M1"]
        self.assertFalse(cfg["early_stopping"])
        self.assertNotIn("lr_schedule", cfg)


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestM1ArchitectureMatchesRealConfig(unittest.TestCase):
    """Built from the real (non-tiny) M0/M1 configs, not synthetic dimensions -
    confirms the parameter-count delta is explained entirely by the extra LSTM
    direction plus the Dense layer's doubled input."""

    def test_parameter_diff_is_explained_by_bidirectional_wrapper_only(self):
        m0 = build_recurrent_model(RecurrentModelConfig.from_experiment("M0"))
        m1 = build_recurrent_model(RecurrentModelConfig.from_experiment("M1"))
        m0_params = count_parameters(m0)["total"]
        m1_params = count_parameters(m1)["total"]

        m0_cfg = RecurrentModelConfig.from_experiment("M0")
        lstm_units, emb_dim = m0_cfg.lstm_units, m0_cfg.embedding_dim
        expected_extra_lstm = 4 * ((emb_dim + lstm_units) * lstm_units + lstm_units)
        expected_extra_dense = 5 * lstm_units  # Dense input doubles (concat fwd+bwd)

        self.assertEqual(m1_params - m0_params, expected_extra_lstm + expected_extra_dense)

    def test_m1_underlying_recurrent_layer_is_lstm_with_no_dropout(self):
        m1 = build_recurrent_model(RecurrentModelConfig.from_experiment("M1"))
        bidir = m1.get_layer("bidirectional_lstm")
        self.assertEqual(bidir.__class__.__name__, "Bidirectional")
        self.assertEqual(bidir.forward_layer.__class__.__name__, "LSTM")
        self.assertEqual(bidir.forward_layer.dropout, 0.0)
        self.assertEqual(bidir.forward_layer.recurrent_dropout, 0.0)

    def test_m1_input_and_output_shape(self):
        m1 = build_recurrent_model(RecurrentModelConfig.from_experiment("M1"))
        self.assertEqual(m1.input_shape, (None, 128))
        self.assertEqual(m1.output_shape, (None, 5))


class TestM1DeltaCalculation(unittest.TestCase):
    """The registered M1 result (project_plan.md §14.9): macro_f1=0.8485154570891954
    against the locked M0 mean 0.8508 -> delta = -0.002284542910804599, unrounded."""

    def test_m1_minus_m0_delta_matches_registered_value(self):
        m1_macro_f1 = 0.8485154570891954
        m0_mean = 0.8508
        deltas = calculate_deltas(current_f1=m1_macro_f1, baseline_m0_f1=m0_mean)
        self.assertAlmostEqual(deltas["delta_vs_m0"], -0.002284542910804599, places=12)

    def test_m1_delta_is_smaller_than_m0_stability_spread(self):
        # project_plan.md §9/§14.7: M0's own three-seed min-max spread is 0.0041 -
        # a delta smaller than that is reported as indistinguishable from
        # run-to-run variance, not as a real effect (stability reference, not a
        # significance test).
        m1_delta = abs(-0.002284542910804599)
        m0_seed_spread = 0.0041
        self.assertLess(m1_delta, m0_seed_spread)


if __name__ == "__main__":
    unittest.main()
