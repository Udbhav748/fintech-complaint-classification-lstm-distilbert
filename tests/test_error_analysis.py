"""Task 16: verifies the M4-vs-D0 error analysis is internally consistent
with the committed prediction files - not a new experiment, not model
testing. Reads results/m4/seed42/test_predictions.csv,
results/d0/seed42/test_predictions.csv, and results/error_analysis_examples.json.
"""

import json
from pathlib import Path
import unittest

import pandas as pd

from src.data import LABELS
from src.evaluation import compute_metrics

M4_PRED = Path("results/m4/seed42/test_predictions.csv")
D0_PRED = Path("results/d0/seed42/test_predictions.csv")
EXAMPLES_JSON = Path("results/error_analysis_examples.json")
M4_METRICS = Path("results/m4/seed42/test_metrics.json")
D0_METRICS = Path("results/d0/seed42/test_metrics.json")


@unittest.skipUnless(M4_PRED.exists() and D0_PRED.exists(), "prediction files not present in this checkout")
class TestPredictionAlignment(unittest.TestCase):
    """Part 2: predictions must be aligned by test_idx, never by row position."""

    def setUp(self):
        self.m4 = pd.read_csv(M4_PRED)
        self.d0 = pd.read_csv(D0_PRED)

    def test_same_test_index_set(self):
        self.assertEqual(set(self.m4["test_idx"]), set(self.d0["test_idx"]))

    def test_same_row_count(self):
        self.assertEqual(len(self.m4), len(self.d0))
        self.assertEqual(len(self.m4), 10181)  # frozen test split size

    def test_y_true_identical_after_alignment(self):
        merged = self.m4.merge(self.d0, on="test_idx", suffixes=("_m4", "_d0"))
        self.assertTrue((merged["y_true_m4"] == merged["y_true_d0"]).all())

    def test_no_duplicate_test_idx(self):
        self.assertEqual(self.m4["test_idx"].nunique(), len(self.m4))
        self.assertEqual(self.d0["test_idx"].nunique(), len(self.d0))


@unittest.skipUnless(M4_PRED.exists() and D0_PRED.exists(), "prediction files not present in this checkout")
class TestConfusionMatrixAndPerClass(unittest.TestCase):
    def setUp(self):
        m4 = pd.read_csv(M4_PRED)
        d0 = pd.read_csv(D0_PRED)
        self.merged = m4.merge(d0, on="test_idx", suffixes=("_m4", "_d0")).rename(
            columns={"y_true_m4": "y_true"}
        ).drop(columns=["y_true_d0"])
        self.m4_metrics = compute_metrics(self.merged["y_true"].values, self.merged["y_pred_m4"].values, labels=LABELS)
        self.d0_metrics = compute_metrics(self.merged["y_true"].values, self.merged["y_pred_d0"].values, labels=LABELS)

    def test_canonical_class_order_used(self):
        self.assertEqual(len(self.m4_metrics["confusion_matrix"]), len(LABELS))
        self.assertEqual(list(self.m4_metrics["per_class"].keys()), LABELS)

    @unittest.skipUnless(M4_METRICS.exists(), "results/m4/seed42/test_metrics.json not present")
    def test_m4_confusion_matrix_matches_committed(self):
        committed = json.loads(M4_METRICS.read_text(encoding="utf-8"))
        self.assertEqual(self.m4_metrics["confusion_matrix"], committed["confusion_matrix"])
        self.assertAlmostEqual(self.m4_metrics["macro_f1"], committed["macro_f1"], places=10)

    @unittest.skipUnless(D0_METRICS.exists(), "results/d0/seed42/test_metrics.json not present")
    def test_d0_confusion_matrix_matches_committed(self):
        committed = json.loads(D0_METRICS.read_text(encoding="utf-8"))
        self.assertEqual(self.d0_metrics["confusion_matrix"], committed["confusion_matrix"])
        self.assertAlmostEqual(self.d0_metrics["macro_f1"], committed["macro_f1"], places=10)

    def test_confusion_pair_counts_match_known_values(self):
        # project_plan.md Task 16: the two largest M4 confusion pairs.
        cm = self.m4_metrics["confusion_matrix"]
        checking_i = LABELS.index("Checking or savings account")
        money_i = LABELS.index("Money transfer, virtual currency, or money service")
        self.assertEqual(cm[money_i][checking_i], 327)
        self.assertEqual(cm[checking_i][money_i], 292)


@unittest.skipUnless(M4_PRED.exists() and D0_PRED.exists(), "prediction files not present in this checkout")
class TestDisagreementGroups(unittest.TestCase):
    def setUp(self):
        m4 = pd.read_csv(M4_PRED)
        d0 = pd.read_csv(D0_PRED)
        self.merged = m4.merge(d0, on="test_idx", suffixes=("_m4", "_d0")).rename(
            columns={"y_true_m4": "y_true"}
        ).drop(columns=["y_true_d0"])
        self.merged["m4_correct"] = self.merged["y_pred_m4"] == self.merged["y_true"]
        self.merged["d0_correct"] = self.merged["y_pred_d0"] == self.merged["y_true"]

    def test_four_groups_sum_to_total(self):
        both_correct = (self.merged["m4_correct"] & self.merged["d0_correct"]).sum()
        both_wrong = (~self.merged["m4_correct"] & ~self.merged["d0_correct"]).sum()
        m4_only = (self.merged["m4_correct"] & ~self.merged["d0_correct"]).sum()
        d0_only = (~self.merged["m4_correct"] & self.merged["d0_correct"]).sum()
        self.assertEqual(both_correct + both_wrong + m4_only + d0_only, len(self.merged))

    def test_group_counts_match_known_values(self):
        both_correct = (self.merged["m4_correct"] & self.merged["d0_correct"]).sum()
        both_wrong = (~self.merged["m4_correct"] & ~self.merged["d0_correct"]).sum()
        m4_only = (self.merged["m4_correct"] & ~self.merged["d0_correct"]).sum()
        d0_only = (~self.merged["m4_correct"] & self.merged["d0_correct"]).sum()
        self.assertEqual(int(both_correct), 8475)
        self.assertEqual(int(both_wrong), 920)
        self.assertEqual(int(m4_only), 364)
        self.assertEqual(int(d0_only), 422)


@unittest.skipUnless(EXAMPLES_JSON.exists(), "results/error_analysis_examples.json not present")
class TestSelectedExamplesTraceability(unittest.TestCase):
    """Part 7/20: selection must be deterministic and every example traceable."""

    def setUp(self):
        self.examples = json.loads(EXAMPLES_JSON.read_text(encoding="utf-8"))
        self.m4 = pd.read_csv(M4_PRED)
        self.d0 = pd.read_csv(D0_PRED)

    def test_every_group_has_at_most_five_examples(self):
        for key, val in self.examples.items():
            self.assertLessEqual(len(val["examples"]), 5)

    def test_examples_sorted_by_test_idx_ascending(self):
        for key, val in self.examples.items():
            indices = [ex["test_idx"] for ex in val["examples"]]
            self.assertEqual(indices, sorted(indices), f"{key} examples must be sorted by test_idx ascending")

    def test_every_selected_index_exists_in_both_prediction_files(self):
        m4_idx = set(self.m4["test_idx"])
        d0_idx = set(self.d0["test_idx"])
        for key, val in self.examples.items():
            for ex in val["examples"]:
                self.assertIn(ex["test_idx"], m4_idx)
                self.assertIn(ex["test_idx"], d0_idx)

    def test_selected_true_labels_match_prediction_files(self):
        m4_lookup = self.m4.set_index("test_idx")
        for key, val in self.examples.items():
            for ex in val["examples"]:
                idx = ex["test_idx"]
                true_code = LABELS.index(ex["true"])
                self.assertEqual(m4_lookup.loc[idx, "y_true"], true_code, f"{key} idx={idx} true-label mismatch")

    def test_excerpts_do_not_contain_full_stop_free_long_digit_runs(self):
        # Sanity check for the masking rule (Part 21): no 6+ digit run should
        # survive in an excerpt (account/phone-like numbers get masked).
        import re
        long_digit = re.compile(r"\d{6,}")
        for key, val in self.examples.items():
            for ex in val["examples"]:
                self.assertIsNone(long_digit.search(ex["excerpt"]), f"unmasked long digit run in {key} idx={ex['test_idx']}")


class TestNoResultsModified(unittest.TestCase):
    """Part 24: this task must not have changed any experiment result."""

    def test_runs_csv_still_has_ten_rows(self):
        df = pd.read_csv("results/runs.csv")
        self.assertEqual(len(df), 10)  # M0x3 + M1 + M2 + M3 + M4 + D0x3

    def test_m4_and_d0_macro_f1_unchanged(self):
        df = pd.read_csv("results/runs.csv")
        m4_f1 = df[(df["experiment"] == "M4") & (df["seed"] == 42)]["macro_f1"].iloc[0]
        d0_f1 = df[(df["experiment"] == "D0") & (df["seed"] == 42)]["macro_f1"].iloc[0]
        self.assertAlmostEqual(m4_f1, 0.8710237463482944, places=12)
        self.assertAlmostEqual(d0_f1, 0.8759140883376707, places=12)


if __name__ == "__main__":
    unittest.main()
