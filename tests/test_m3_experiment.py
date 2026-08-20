"""M3-specific integration checks (project_plan.md Stage 9).

Generic config/schema/delta/traceability behavior is already covered by
tests/test_config.py, tests/test_results.py, and tests/test_models.py (which
already builds M3 via the factory and checks GloVe matrix loading/shape
rejection). This file checks the one thing those don't: that M3's registered
configuration is *exactly* M2 plus the embedding source - no other field
silently drifted - that the frozen GloVe artifact actually matches the
expected shape, and that the registered result's deltas match the locked
M0/M2 values.
"""

from pathlib import Path
import unittest

from src.config import EXPERIMENT_CONFIGS, get_experiment_config
from src.evaluation import calculate_deltas

try:
    from src.models import RecurrentModelConfig, build_recurrent_model, count_parameters

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TestM3IsOnlyEmbeddingSourceChange(unittest.TestCase):
    """project_plan.md §14.11: 'The only intended change from M2 is the
    embedding source.'"""

    ALLOWED_DIFFERENCES = {"experiment", "model_type", "description", "seeds"}
    REQUIRED_DIFFERENCES = {"use_glove", "glove_path"}

    def test_only_embedding_source_and_bookkeeping_fields_differ(self):
        m2 = get_experiment_config("M2")
        m3 = get_experiment_config("M3")
        all_keys = set(m2) | set(m3)
        differing = {k for k in all_keys if m2.get(k) != m3.get(k)}
        unexpected = differing - self.ALLOWED_DIFFERENCES - self.REQUIRED_DIFFERENCES
        self.assertEqual(unexpected, set(), f"M2->M3 changed unexpected fields: {unexpected}")
        self.assertTrue(
            self.REQUIRED_DIFFERENCES.issubset(differing),
            f"M2->M3 must change {self.REQUIRED_DIFFERENCES}, only changed {differing}",
        )

    def test_m3_keeps_m2_regularization_unchanged(self):
        cfg = EXPERIMENT_CONFIGS["M3"]
        self.assertEqual(cfg["dropout"], 0.3)
        self.assertEqual(cfg["recurrent_dropout"], 0.2)
        self.assertEqual(cfg["spatial_dropout"], 0.2)

    def test_m3_uses_glove_and_is_trainable(self):
        cfg = EXPERIMENT_CONFIGS["M3"]
        self.assertTrue(cfg["use_glove"])
        self.assertTrue(cfg["glove_trainable"])

    def test_m3_still_bidirectional_max_len_128(self):
        cfg = EXPERIMENT_CONFIGS["M3"]
        self.assertTrue(cfg["bidirectional"])
        self.assertEqual(cfg["max_len"], 128)

    def test_m3_seed_policy(self):
        self.assertEqual(EXPERIMENT_CONFIGS["M3"]["seeds"], [42])  # single-seed rung

    def test_m3_no_early_stopping_or_lr_schedule(self):
        cfg = EXPERIMENT_CONFIGS["M3"]
        self.assertFalse(cfg["early_stopping"])
        self.assertNotIn("lr_schedule", cfg)


class TestFrozenGloveArtifact(unittest.TestCase):
    """The matrix M3 actually trained with - shipped to Kaggle as-is, never
    rebuilt there."""

    def test_matrix_file_exists_and_has_expected_shape(self):
        import numpy as np

        matrix_path = Path("artifacts/embeddings/glove_100d_matrix.npy")
        self.assertTrue(matrix_path.exists())
        matrix = np.load(matrix_path)
        cfg = EXPERIMENT_CONFIGS["M3"]
        self.assertEqual(matrix.shape, (cfg["vocab_size"], cfg["embedding_dim"]))

    def test_padding_row_is_zero(self):
        import numpy as np

        matrix = np.load("artifacts/embeddings/glove_100d_matrix.npy")
        self.assertTrue(np.all(matrix[0] == 0.0))


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestM3ArchitectureMatchesRealConfig(unittest.TestCase):
    """Built from the real (non-tiny) M2/M3 configs and the real frozen
    matrix, not synthetic dimensions."""

    def _load_matrix(self):
        import numpy as np

        return np.load("artifacts/embeddings/glove_100d_matrix.npy")

    def test_parameter_count_identical_to_m2(self):
        m2 = build_recurrent_model(RecurrentModelConfig.from_experiment("M2"))
        m3 = build_recurrent_model(RecurrentModelConfig.from_experiment("M3"), embedding_matrix=self._load_matrix())
        self.assertEqual(count_parameters(m2)["total"], count_parameters(m3)["total"])
        self.assertEqual(count_parameters(m2)["trainable"], count_parameters(m3)["trainable"])

    def test_embedding_weights_match_frozen_matrix(self):
        import numpy as np

        matrix = self._load_matrix()
        m3 = build_recurrent_model(RecurrentModelConfig.from_experiment("M3"), embedding_matrix=matrix)
        loaded = m3.get_layer("embedding").get_weights()[0]
        np.testing.assert_allclose(loaded, matrix)

    def test_m3_lstm_dropout_and_recurrent_dropout_unchanged_from_m2(self):
        m3 = build_recurrent_model(RecurrentModelConfig.from_experiment("M3"), embedding_matrix=self._load_matrix())
        bidir = m3.get_layer("bidirectional_lstm")
        self.assertEqual(bidir.forward_layer.__class__.__name__, "LSTM")
        self.assertAlmostEqual(bidir.forward_layer.dropout, 0.3)
        self.assertAlmostEqual(bidir.forward_layer.recurrent_dropout, 0.2)

    def test_m3_input_and_output_shape(self):
        m3 = build_recurrent_model(RecurrentModelConfig.from_experiment("M3"), embedding_matrix=self._load_matrix())
        self.assertEqual(m3.input_shape, (None, 128))
        self.assertEqual(m3.output_shape, (None, 5))


class TestM3DeltaCalculation(unittest.TestCase):
    """The registered M3 result (project_plan.md §14.11):
    macro_f1=0.8656677597625515 against M2=0.8517974977710514 and
    M0 mean=0.8508."""

    def test_m3_minus_m2_delta_matches_registered_value(self):
        m3_macro_f1 = 0.8656677597625515
        m2_macro_f1 = 0.8517974977710514
        deltas = calculate_deltas(current_f1=m3_macro_f1, baseline_m0_f1=m2_macro_f1)
        self.assertAlmostEqual(deltas["delta_vs_m0"], 0.013870261991500099, places=12)

    def test_m3_minus_m0_delta_matches_registered_value(self):
        m3_macro_f1 = 0.8656677597625515
        m0_mean = 0.8508
        deltas = calculate_deltas(current_f1=m3_macro_f1, baseline_m0_f1=m0_mean)
        self.assertAlmostEqual(deltas["delta_vs_m0"], 0.014867759762551502, places=12)

    def test_m3_minus_m2_delta_exceeds_m0_stability_spread(self):
        # project_plan.md §9: M0's own three-seed min-max spread is 0.0041.
        # Unlike M1 and M2, M3's delta is large enough to be a resolved effect.
        m3_m2_delta = abs(0.013870261991500099)
        m0_seed_spread = 0.0041
        self.assertGreater(m3_m2_delta, m0_seed_spread)


if __name__ == "__main__":
    unittest.main()
