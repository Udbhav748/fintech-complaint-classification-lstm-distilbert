"""Unit tests for the recurrent model factory (M0-M4).

Requires TensorFlow/Keras, unlike the rest of the test suite. If it is not
installed in the environment running pytest, every test here is skipped (not
failed) - `src.models` is the one module in this project that genuinely needs a
deep learning framework, and the rest of the suite must stay runnable without it.

Uses tiny synthetic dimensions throughout (Part 19: "do not train full models in
unit tests"), never the real 20,000-word/128-unit production config - these tests
check architecture shape and configuration wiring, not the actual M0-M4 numbers,
which come from `src.config.EXPERIMENT_CONFIGS` and are checked separately below.
"""

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from src.checkpoint import get_checkpoint_path
from src.data import LABELS
from src.reproducibility import set_seed

try:
    from src.evaluation import compute_val_macro_f1
    from src.models import (
        MacroF1Score,
        ModelConfigError,
        RecurrentModelConfig,
        build_callbacks,
        build_recurrent_model,
        count_parameters,
    )

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def tiny_config(experiment_id: str, **overrides) -> "RecurrentModelConfig":
    base = dict(
        experiment_id=experiment_id, vocab_size=50, embedding_dim=8, embedding_type="random",
        embedding_trainable=True, max_len=10, lstm_units=4, bidirectional=False,
        dropout=0.0, recurrent_dropout=0.0, spatial_dropout=0.0, output_classes=len(LABELS),
        learning_rate=1e-3, optimizer="adam",
    )
    base.update(overrides)
    return RecurrentModelConfig(**base)


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestConfigFromExperiment(unittest.TestCase):
    """Part 2: the config comes from src.config.EXPERIMENT_CONFIGS, not invented."""

    def test_all_five_recurrent_experiments_load(self):
        for exp in ["M0", "M1", "M2", "M3", "M4"]:
            cfg = RecurrentModelConfig.from_experiment(exp)
            self.assertEqual(cfg.experiment_id, exp)
            self.assertEqual(cfg.output_classes, 5)

    def test_d0_is_rejected(self):
        with self.assertRaises(ModelConfigError):
            RecurrentModelConfig.from_experiment("D0")

    def test_recurrent_dropout_locked_at_0_2_for_m2_m3_m4(self):
        for exp in ["M2", "M3", "M4"]:
            self.assertEqual(RecurrentModelConfig.from_experiment(exp).recurrent_dropout, 0.2)

    def test_m4_lr_schedule_locked_values(self):
        cfg = RecurrentModelConfig.from_experiment("M4")
        self.assertEqual(cfg.lr_schedule_factor, 0.5)
        self.assertEqual(cfg.lr_schedule_patience, 2)
        self.assertEqual(cfg.lr_schedule_min_lr, 1e-5)
        self.assertEqual(cfg.early_stopping_patience, 3)


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestConfigValidation(unittest.TestCase):
    """Part 13: fail loudly before Keras does."""

    def test_rejects_non_recurrent_experiment_id(self):
        with self.assertRaises(ModelConfigError):
            tiny_config("D0")

    def test_rejects_zero_vocab_size(self):
        with self.assertRaises(ModelConfigError):
            tiny_config("M0", vocab_size=0)

    def test_rejects_zero_max_len(self):
        with self.assertRaises(ModelConfigError):
            tiny_config("M0", max_len=0)

    def test_rejects_zero_lstm_units(self):
        with self.assertRaises(ModelConfigError):
            tiny_config("M0", lstm_units=0)

    def test_rejects_wrong_output_classes(self):
        with self.assertRaises(ModelConfigError):
            tiny_config("M0", output_classes=3)

    def test_rejects_invalid_embedding_type(self):
        with self.assertRaises(ModelConfigError):
            tiny_config("M0", embedding_type="word2vec")

    def test_rejects_out_of_range_dropout(self):
        with self.assertRaises(ModelConfigError):
            tiny_config("M0", dropout=1.5)

    def test_rejects_negative_learning_rate(self):
        with self.assertRaises(ModelConfigError):
            tiny_config("M0", learning_rate=-0.1)


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestArchitecturePerExperiment(unittest.TestCase):
    """Part 14: inspect architecture, do not train."""

    def _layer_types(self, model):
        return [layer.__class__.__name__ for layer in model.layers]

    def test_m0_is_exactly_input_embedding_lstm_dense(self):
        model = build_recurrent_model(tiny_config("M0"))
        self.assertEqual(self._layer_types(model), ["InputLayer", "Embedding", "LSTM", "Dense"])

    def test_m0_output_layer_has_5_units(self):
        model = build_recurrent_model(tiny_config("M0"))
        self.assertEqual(model.layers[-1].units, 5)

    def test_m0_lstm_has_no_dropout(self):
        model = build_recurrent_model(tiny_config("M0"))
        lstm = model.get_layer("lstm")
        self.assertEqual(lstm.dropout, 0.0)
        self.assertEqual(lstm.recurrent_dropout, 0.0)

    def test_m1_wraps_lstm_in_bidirectional(self):
        model = build_recurrent_model(tiny_config("M1", bidirectional=True))
        self.assertEqual(self._layer_types(model), ["InputLayer", "Embedding", "Bidirectional", "Dense"])

    def test_m1_has_same_embedding_shape_as_m0(self):
        m0 = build_recurrent_model(tiny_config("M0"))
        m1 = build_recurrent_model(tiny_config("M1", bidirectional=True))
        self.assertEqual(m0.get_layer("embedding").output.shape, m1.get_layer("embedding").output.shape)

    def test_m2_has_spatial_dropout_and_recurrent_dropout(self):
        model = build_recurrent_model(
            tiny_config("M2", bidirectional=True, dropout=0.3, recurrent_dropout=0.2, spatial_dropout=0.2)
        )
        self.assertEqual(
            self._layer_types(model), ["InputLayer", "Embedding", "SpatialDropout1D", "Bidirectional", "Dense"]
        )
        inner_lstm = model.get_layer("bidirectional_lstm").forward_layer
        self.assertAlmostEqual(inner_lstm.dropout, 0.3)
        self.assertAlmostEqual(inner_lstm.recurrent_dropout, 0.2)

    def test_m2_has_no_stacked_lstm(self):
        model = build_recurrent_model(
            tiny_config("M2", bidirectional=True, dropout=0.3, recurrent_dropout=0.2, spatial_dropout=0.2)
        )
        lstm_like = [l for l in model.layers if "lstm" in l.name.lower()]
        self.assertEqual(len(lstm_like), 1, "M2 must not have stacked recurrent layers")

    def test_m3_same_recurrent_architecture_as_m2_only_embedding_differs(self):
        m2 = build_recurrent_model(
            tiny_config("M2", bidirectional=True, dropout=0.3, recurrent_dropout=0.2, spatial_dropout=0.2)
        )
        rng = np.random.default_rng(0)
        matrix = rng.uniform(-0.05, 0.05, size=(50, 8)).astype(np.float32)
        m3 = build_recurrent_model(
            tiny_config("M3", bidirectional=True, dropout=0.3, recurrent_dropout=0.2, spatial_dropout=0.2,
                        embedding_type="glove"),
            embedding_matrix=matrix,
        )
        self.assertEqual(self._layer_types(m2), self._layer_types(m3))
        self.assertEqual(m2.get_layer("embedding").output.shape, m3.get_layer("embedding").output.shape)

    def test_m3_embedding_matrix_is_actually_loaded(self):
        rng = np.random.default_rng(0)
        matrix = rng.uniform(-0.05, 0.05, size=(50, 8)).astype(np.float32)
        m3 = build_recurrent_model(
            tiny_config("M3", bidirectional=True, embedding_type="glove"), embedding_matrix=matrix
        )
        loaded = m3.get_layer("embedding").get_weights()[0]
        np.testing.assert_allclose(loaded, matrix)

    def test_m4_same_architecture_as_m3_max_len_256_is_the_only_shape_change(self):
        m3 = build_recurrent_model(
            tiny_config("M3", bidirectional=True, dropout=0.3, recurrent_dropout=0.2, spatial_dropout=0.2,
                        embedding_type="glove", max_len=10),
            embedding_matrix=np.zeros((50, 8), dtype=np.float32),
        )
        m4 = build_recurrent_model(
            tiny_config("M4", bidirectional=True, dropout=0.3, recurrent_dropout=0.2, spatial_dropout=0.2,
                        embedding_type="glove", max_len=20),
            embedding_matrix=np.zeros((50, 8), dtype=np.float32),
        )
        self.assertEqual(self._layer_types(m3), self._layer_types(m4))
        self.assertEqual(m3.input_shape, (None, 10))
        self.assertEqual(m4.input_shape, (None, 20))


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestEmbeddingValidation(unittest.TestCase):
    """Part 4: GloVe matrix comes from preprocessing, is never rebuilt here."""

    def test_glove_requires_a_matrix(self):
        with self.assertRaises(ModelConfigError):
            build_recurrent_model(tiny_config("M3", embedding_type="glove"))

    def test_glove_matrix_wrong_shape_rejected(self):
        wrong = np.zeros((50, 16), dtype=np.float32)  # dim mismatch: config says 8
        with self.assertRaises(ModelConfigError):
            build_recurrent_model(tiny_config("M3", embedding_type="glove"), embedding_matrix=wrong)

    def test_random_embedding_with_matrix_is_rejected(self):
        matrix = np.zeros((50, 8), dtype=np.float32)
        with self.assertRaises(ModelConfigError):
            build_recurrent_model(tiny_config("M0", embedding_type="random"), embedding_matrix=matrix)


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestCallbacks(unittest.TestCase):
    """Part 12: configuration only, nothing is trained."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_m0_has_only_checkpoint_callback(self):
        cbs = build_callbacks(tiny_config("M0"), self.tmp / "M0.weights.h5")
        self.assertEqual([c.__class__.__name__ for c in cbs], ["ModelCheckpoint"])

    def test_m4_has_checkpoint_scheduler_and_early_stopping(self):
        cfg = tiny_config(
            "M4", lr_schedule="reduce_on_plateau", lr_schedule_factor=0.5,
            lr_schedule_patience=2, lr_schedule_min_lr=1e-5,
            early_stopping=True, early_stopping_patience=3,
        )
        cbs = build_callbacks(cfg, self.tmp / "M4.weights.h5")
        self.assertEqual(
            [c.__class__.__name__ for c in cbs],
            ["ModelCheckpoint", "ReduceLROnPlateau", "EarlyStopping"],
        )

    def test_checkpoint_callback_monitors_val_macro_f1_and_saves_best_only(self):
        cbs = build_callbacks(tiny_config("M0"), self.tmp / "M0.weights.h5")
        ckpt = cbs[0]
        self.assertEqual(ckpt.monitor, "val_macro_f1")
        self.assertTrue(ckpt.save_best_only)

    def test_scheduler_missing_params_fails_loudly(self):
        cfg = tiny_config("M4", lr_schedule="reduce_on_plateau")  # factor/patience/min_lr left None
        with self.assertRaises(ModelConfigError):
            build_callbacks(cfg, self.tmp / "M4.weights.h5")

    def test_early_stopping_missing_patience_fails_loudly(self):
        cfg = tiny_config("M4", early_stopping=True)  # patience left None
        with self.assertRaises(ModelConfigError):
            build_callbacks(cfg, self.tmp / "M4.weights.h5")


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestMacroF1Metric(unittest.TestCase):
    """Part 7: val_macro_f1 must equal sklearn macro-F1 on the full validation
    set - checked here via genuine Keras Metric accumulation, not a per-batch
    average."""

    def test_matches_sklearn_across_simulated_batches(self):
        rng = np.random.default_rng(11)
        y_true = rng.integers(0, 5, size=83)
        y_pred = y_true.copy()
        flip = rng.random(83) < 0.35
        y_pred[flip] = rng.integers(0, 5, size=int(flip.sum()))
        probs = np.zeros((83, 5), dtype=np.float32)
        probs[np.arange(83), y_pred] = 1.0

        metric = MacroF1Score(num_classes=5)
        for i in range(0, 83, 7):  # deliberately uneven batch size
            metric.update_state(y_true[i:i + 7], probs[i:i + 7])
        keras_val = float(metric.result().numpy())

        sklearn_val = compute_val_macro_f1(y_true, y_pred, labels=LABELS)
        self.assertAlmostEqual(keras_val, sklearn_val, places=5)

    def test_reset_state_clears_accumulator(self):
        metric = MacroF1Score(num_classes=5)
        y = np.array([0, 1, 2, 3, 4])
        probs = np.eye(5, dtype=np.float32)
        metric.update_state(y, probs)
        self.assertAlmostEqual(float(metric.result().numpy()), 1.0)
        metric.reset_state()
        np.testing.assert_array_equal(metric.confusion.numpy(), np.zeros((5, 5)))


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestParameterCount(unittest.TestCase):
    def test_returns_positive_trainable_count(self):
        model = build_recurrent_model(tiny_config("M0"))
        counts = count_parameters(model)
        self.assertGreater(counts["trainable"], 0)
        self.assertEqual(counts["non_trainable"], 0)
        self.assertEqual(counts["total"], counts["trainable"] + counts["non_trainable"])

    def test_frozen_embedding_moves_params_to_non_trainable(self):
        matrix = np.zeros((50, 8), dtype=np.float32)
        model = build_recurrent_model(
            tiny_config("M3", embedding_type="glove", embedding_trainable=False), embedding_matrix=matrix
        )
        counts = count_parameters(model)
        self.assertGreater(counts["non_trainable"], 0)

    def test_larger_model_has_more_parameters(self):
        small = count_parameters(build_recurrent_model(tiny_config("M0", lstm_units=4)))
        large = count_parameters(build_recurrent_model(tiny_config("M0", lstm_units=16)))
        self.assertGreater(large["trainable"], small["trainable"])


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestCheckpointCompatibility(unittest.TestCase):
    """Part 18: use the existing checkpoint utilities, don't build a new one."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_and_load_weights_round_trip(self):
        ckpt_path = get_checkpoint_path("M0", seed=42, checkpoint_dir=self.tmp)
        model = build_recurrent_model(tiny_config("M0"))
        model.save_weights(str(ckpt_path))
        self.assertTrue(ckpt_path.exists())
        self.assertGreater(ckpt_path.stat().st_size, 0)

        reloaded = build_recurrent_model(tiny_config("M0"))
        reloaded.load_weights(str(ckpt_path))
        for w1, w2 in zip(model.weights, reloaded.weights):
            np.testing.assert_allclose(w1.numpy(), w2.numpy())


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestReproducibility(unittest.TestCase):
    """Part 16: same config + same seed -> same initial weights. CPU-only here;
    this does not claim GPU/cuDNN-level bitwise determinism, which TensorFlow
    does not guarantee even with a fixed seed (documented TF/Kaggle limitation)."""

    def test_same_seed_gives_identical_initial_weights(self):
        set_seed(42)
        a = build_recurrent_model(tiny_config("M0"))
        set_seed(42)
        b = build_recurrent_model(tiny_config("M0"))
        for w1, w2 in zip(a.weights, b.weights):
            np.testing.assert_allclose(w1.numpy(), w2.numpy())

    def test_different_seed_gives_different_initial_weights(self):
        set_seed(42)
        a = build_recurrent_model(tiny_config("M0"))
        set_seed(123)
        b = build_recurrent_model(tiny_config("M0"))
        differs = any(not np.allclose(w1.numpy(), w2.numpy()) for w1, w2 in zip(a.weights, b.weights))
        self.assertTrue(differs)

    def test_same_config_gives_same_parameter_shapes(self):
        a = build_recurrent_model(tiny_config("M2", bidirectional=True, dropout=0.3, recurrent_dropout=0.2, spatial_dropout=0.2))
        b = build_recurrent_model(tiny_config("M2", bidirectional=True, dropout=0.3, recurrent_dropout=0.2, spatial_dropout=0.2))
        self.assertEqual([w.shape for w in a.weights], [w.shape for w in b.weights])


@unittest.skipUnless(TF_AVAILABLE, "tensorflow/keras not installed in this environment")
class TestNoExcludedArchitecture(unittest.TestCase):
    """Part: stacked LSTM and class weights are explicitly excluded project-wide."""

    def test_no_experiment_config_has_stacked_layers(self):
        for exp in ["M0", "M1", "M2", "M3", "M4"]:
            cfg = RecurrentModelConfig.from_experiment(exp)
            matrix = None
            if cfg.embedding_type == "glove":
                matrix = np.zeros((cfg.vocab_size, cfg.embedding_dim), dtype=np.float32)
            # Real configs are large (20k vocab, max_len 128/256); only check
            # layer composition, not build a full-size model in a unit test.
            small = tiny_config(
                exp, embedding_type=cfg.embedding_type, bidirectional=cfg.bidirectional,
                dropout=cfg.dropout, recurrent_dropout=cfg.recurrent_dropout,
                spatial_dropout=cfg.spatial_dropout,
            )
            small_matrix = np.zeros((50, 8), dtype=np.float32) if cfg.embedding_type == "glove" else None
            model = build_recurrent_model(small, embedding_matrix=small_matrix)
            lstm_like = [l for l in model.layers if "lstm" in l.name.lower()]
            self.assertEqual(len(lstm_like), 1, f"{exp} must have exactly one recurrent layer, not stacked")


if __name__ == "__main__":
    unittest.main()
