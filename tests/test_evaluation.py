"""Unit tests for the evaluation layer and metric calculations.

Explicitly validates:
- Mathematical equivalence: compute_metrics Macro-F1 == sklearn f1_score(average='macro')
- Canonical class ordering preservation
- Metric dictionary output schema and shapes
- Delta calculation accuracy and explicit sign conventions
- Edge cases (division by zero, single-class predictions, zero-support classes)
"""

import unittest

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data import LABELS
from src.evaluation import (
    calculate_deltas,
    compute_metrics,
    compute_val_macro_f1,
    format_delta,
    seed_statistics,
)


class TestEvaluationLayer(unittest.TestCase):
    def setUp(self):
        self.labels = LABELS  # 5 canonical classes
        self.num_classes = len(LABELS)

    def test_equivalence_with_scikit_learn(self):
        """Crucial test: custom evaluation layer matches scikit-learn exactly."""
        np.random.seed(42)
        n_samples = 500
        y_true = np.random.randint(0, self.num_classes, size=n_samples)
        # Add some noise to generate realistic imperfect predictions
        y_pred = y_true.copy()
        flip_mask = np.random.rand(n_samples) < 0.25
        y_pred[flip_mask] = np.random.randint(0, self.num_classes, size=int(flip_mask.sum()))

        res = compute_metrics(y_true, y_pred, labels=self.labels)

        # Direct sklearn reference calculations
        expected_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        expected_acc = accuracy_score(y_true, y_pred)
        expected_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        expected_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        expected_cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))

        self.assertAlmostEqual(res["macro_f1"], expected_f1, places=7)
        self.assertAlmostEqual(res["accuracy"], expected_acc, places=7)
        self.assertAlmostEqual(res["macro_precision"], expected_prec, places=7)
        self.assertAlmostEqual(res["macro_recall"], expected_rec, places=7)
        np.testing.assert_array_equal(res["confusion_matrix"], expected_cm)

    def test_class_ordering_and_per_class_metrics(self):
        """Verifies that all 5 classes appear in exact canonical order."""
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 4]
        res = compute_metrics(y_true, y_pred, labels=self.labels)

        per_class = res["per_class"]
        self.assertEqual(list(per_class.keys()), self.labels)

        for label in self.labels:
            self.assertEqual(per_class[label]["f1"], 1.0)
            self.assertEqual(per_class[label]["precision"], 1.0)
            self.assertEqual(per_class[label]["recall"], 1.0)
            self.assertEqual(per_class[label]["support"], 1)

    def test_metric_output_shape_and_keys(self):
        """Verifies dictionary structure, matrix dimensions, and types."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 4, 1, 0])
        res = compute_metrics(y_true, y_pred, labels=self.labels)

        self.assertIn("accuracy", res)
        self.assertIn("macro_f1", res)
        self.assertIn("macro_precision", res)
        self.assertIn("macro_recall", res)
        self.assertIn("per_class", res)
        self.assertIn("confusion_matrix", res)

        cm = np.array(res["confusion_matrix"])
        self.assertEqual(cm.shape, (5, 5))
        self.assertEqual(len(res["per_class"]), 5)

    def test_delta_calculations_and_sign_conventions(self):
        """Verifies delta calculations with unrounded math and presentation formatting."""
        # Baseline M0
        m0_f1 = 0.84201234
        # M1 improved over M0
        m1_f1 = 0.85805678
        # M2 declined vs M1
        m2_f1 = 0.85301111

        # Check M0 (baseline)
        deltas_m0 = calculate_deltas(current_f1=m0_f1, previous_f1=None, baseline_m0_f1=None)
        self.assertIsNone(deltas_m0["delta_vs_previous"])
        self.assertIsNone(deltas_m0["delta_vs_m0"])
        self.assertEqual(format_delta(deltas_m0["delta_vs_previous"]), "—")
        self.assertEqual(format_delta(deltas_m0["delta_vs_m0"]), "—")

        # Check M1 (improvement)
        deltas_m1 = calculate_deltas(current_f1=m1_f1, previous_f1=m0_f1, baseline_m0_f1=m0_f1)
        expected_delta_prev = m1_f1 - m0_f1
        self.assertAlmostEqual(deltas_m1["delta_vs_previous"], expected_delta_prev, places=8)
        self.assertEqual(format_delta(deltas_m1["delta_vs_previous"]), "+0.0160")
        self.assertEqual(format_delta(deltas_m1["delta_vs_m0"]), "+0.0160")

        # Check M2 (decline vs previous, positive vs M0)
        deltas_m2 = calculate_deltas(current_f1=m2_f1, previous_f1=m1_f1, baseline_m0_f1=m0_f1)
        self.assertAlmostEqual(deltas_m2["delta_vs_previous"], m2_f1 - m1_f1, places=8)
        self.assertAlmostEqual(deltas_m2["delta_vs_m0"], m2_f1 - m0_f1, places=8)
        self.assertEqual(format_delta(deltas_m2["delta_vs_previous"]), "-0.0050")
        self.assertEqual(format_delta(deltas_m2["delta_vs_m0"]), "+0.0110")

    def test_edge_cases_single_class_and_zero_division(self):
        """Verifies handling of extreme cases without crashing or producing NaN."""
        # All predictions belong to class 0
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 0, 0, 0, 0])

        res = compute_metrics(y_true, y_pred, labels=self.labels)
        self.assertFalse(np.isnan(res["macro_f1"]))
        self.assertFalse(np.isnan(res["accuracy"]))
        self.assertEqual(res["accuracy"], 0.20)

        # Sklearn macro-F1 on this case: class 0 has P=0.2, R=1.0, F1=0.3333; others have F1=0.0
        expected_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        self.assertAlmostEqual(res["macro_f1"], expected_f1, places=7)


class TestSyntheticEquivalenceCases(unittest.TestCase):
    """Each case in Task 4's Part 2 list, checked directly against sklearn."""

    def setUp(self):
        self.labels = LABELS
        self.num_classes = len(LABELS)

    def _assert_matches_sklearn(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        res = compute_metrics(y_true, y_pred, labels=self.labels)
        self.assertAlmostEqual(
            res["macro_f1"], f1_score(y_true, y_pred, average="macro", zero_division=0), places=7
        )
        self.assertAlmostEqual(
            res["macro_precision"], precision_score(y_true, y_pred, average="macro", zero_division=0), places=7
        )
        self.assertAlmostEqual(
            res["macro_recall"], recall_score(y_true, y_pred, average="macro", zero_division=0), places=7
        )
        self.assertAlmostEqual(res["accuracy"], accuracy_score(y_true, y_pred), places=7)
        return res

    def test_balanced_correct_predictions(self):
        y = [0, 1, 2, 3, 4] * 10
        res = self._assert_matches_sklearn(y, y)
        self.assertEqual(res["macro_f1"], 1.0)

    def test_one_class_with_poor_recall(self):
        # Class 2 (Debt collection) is mostly misclassified as class 0; every
        # other class is predicted perfectly.
        y_true = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10
        y_pred = [0] * 10 + [1] * 10 + ([0] * 8 + [2] * 2) + [3] * 10 + [4] * 10
        self._assert_matches_sklearn(y_true, y_pred)

    def test_highly_uneven_predictions(self):
        # 90 examples of class 0, a handful of everything else, imperfect predictions.
        y_true = [0] * 90 + [1] * 3 + [2] * 3 + [3] * 2 + [4] * 2
        y_pred = [0] * 85 + [1] * 5 + [1] * 2 + [2] * 1 + [3] * 3 + [4] * 1 + [0] * 3
        self._assert_matches_sklearn(y_true[: len(y_pred)], y_pred)

    def test_missing_predicted_class(self):
        # Class 4 exists in y_true but the model never predicts it.
        y_true = [0, 1, 2, 3, 4, 4, 4]
        y_pred = [0, 1, 2, 3, 3, 3, 2]
        res = self._assert_matches_sklearn(y_true, y_pred)
        self.assertEqual(res["per_class"][self.labels[4]]["f1"], 0.0)
        self.assertEqual(res["per_class"][self.labels[4]]["support"], 3)

    def test_missing_true_class_zero_support(self):
        # Class 3 never occurs in y_true (zero support) but is predicted once.
        y_true = [0, 1, 2, 4, 4]
        y_pred = [0, 1, 2, 3, 4]
        res = self._assert_matches_sklearn(y_true, y_pred)
        self.assertEqual(res["per_class"][self.labels[3]]["support"], 0)
        self.assertFalse(np.isnan(res["macro_f1"]))

    def test_all_predictions_one_class(self):
        y_true = [0, 1, 2, 3, 4] * 4
        y_pred = [2] * 20
        self._assert_matches_sklearn(y_true, y_pred)


class TestInputValidation(unittest.TestCase):
    """Part 11/12: invalid input must fail loudly, not produce silent nonsense."""

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            compute_metrics([0, 1, 2], [0, 1], labels=LABELS)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            compute_metrics([], [], labels=LABELS)

    def test_rejects_out_of_range_integer_label(self):
        # Only 5 classes (ids 0-4); 5 is out of range and must not be silently
        # dropped from the macro average.
        with self.assertRaises(ValueError):
            compute_metrics([0, 1, 5], [0, 1, 2], labels=LABELS)

    def test_rejects_unrecognized_string_label(self):
        with self.assertRaises(ValueError):
            compute_metrics(["not a real product"], [LABELS[0]], labels=LABELS)

    def test_rejects_mixed_int_and_string_labels(self):
        with self.assertRaises(ValueError):
            compute_metrics([0, 1, 2], [LABELS[0], LABELS[1], LABELS[2]], labels=LABELS)

    def test_rejects_non_1d_input(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.array([[0, 1], [2, 3]]), np.array([[0, 1], [2, 3]]), labels=LABELS)


class TestConfusionMatrixPositions(unittest.TestCase):
    def test_matrix_positions_match_canonical_order(self):
        # 3 true=0 predicted=0, 2 true=1 predicted=2, 1 true=4 predicted=4.
        y_true = [0, 0, 0, 1, 1, 4]
        y_pred = [0, 0, 0, 2, 2, 4]
        res = compute_metrics(y_true, y_pred, labels=LABELS)
        cm = np.array(res["confusion_matrix"])
        self.assertEqual(cm.shape, (5, 5))
        self.assertEqual(cm[0, 0], 3)  # true=0 -> pred=0
        self.assertEqual(cm[1, 2], 2)  # true=1 -> pred=2
        self.assertEqual(cm[4, 4], 1)  # true=4 -> pred=4
        self.assertEqual(cm.sum(), 6)


class TestSeedStatistics(unittest.TestCase):
    def test_matches_hand_computed_mean_and_sample_std(self):
        values = [0.841, 0.843, 0.842]
        stats = seed_statistics(values)
        self.assertAlmostEqual(stats["mean"], np.mean(values), places=10)
        self.assertAlmostEqual(stats["std"], np.std(values, ddof=1), places=10)
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["ddof"], 1)

    def test_single_value_std_is_none_not_zero(self):
        stats = seed_statistics([0.858])
        self.assertIsNone(stats["std"])
        self.assertEqual(stats["n"], 1)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            seed_statistics([])


class TestValMacroF1Checkpointing(unittest.TestCase):
    """Part 7: val_macro_f1 must equal a full-validation-set sklearn Macro-F1,
    and must NOT be computed as a running average of per-batch macro-F1."""

    def test_matches_manual_sklearn_on_synthetic_validation_predictions(self):
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, len(LABELS), size=200)
        y_pred = y_true.copy()
        flip = rng.random(200) < 0.3
        y_pred[flip] = rng.integers(0, len(LABELS), size=int(flip.sum()))

        val_f1 = compute_val_macro_f1(y_true, y_pred, labels=LABELS)
        expected = f1_score(y_true, y_pred, average="macro", zero_division=0)
        self.assertAlmostEqual(val_f1, expected, places=7)

    def test_per_batch_averaging_would_be_wrong(self):
        """Demonstrates why val_macro_f1 must be computed on the full validation
        set rather than averaged batch-by-batch, per Part 7's explicit warning."""
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, len(LABELS), size=100)
        y_pred = y_true.copy()
        flip = rng.random(100) < 0.4
        y_pred[flip] = rng.integers(0, len(LABELS), size=int(flip.sum()))

        correct_full_set = compute_val_macro_f1(y_true, y_pred, labels=LABELS)

        # Naive (incorrect) pattern: compute macro-F1 per batch, then average.
        batch_size = 10
        batch_scores = [
            compute_val_macro_f1(y_true[i : i + batch_size], y_pred[i : i + batch_size], labels=LABELS)
            for i in range(0, len(y_true), batch_size)
        ]
        naive_batch_averaged = float(np.mean(batch_scores))

        self.assertNotAlmostEqual(correct_full_set, naive_batch_averaged, places=2)


if __name__ == "__main__":
    unittest.main()
