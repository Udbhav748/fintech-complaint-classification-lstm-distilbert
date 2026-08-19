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
from src.evaluation import calculate_deltas, compute_metrics, format_delta


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


if __name__ == "__main__":
    unittest.main()
