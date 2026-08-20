"""Task 12: verifies the recurrent-ladder selection analysis, not a new
experiment. Reads the same committed results/runs.csv every prior stage
wrote to and checks that the derived decision artifact
(results/best_recurrent.json) is internally consistent with it - this is
analysis-integrity testing, not model testing.
"""

import json
from pathlib import Path
import unittest

import pandas as pd

from src.evaluation import calculate_deltas, seed_statistics

RUNS_CSV = Path("results/runs.csv")
BEST_RECURRENT_JSON = Path("results/best_recurrent.json")


@unittest.skipUnless(RUNS_CSV.exists(), "results/runs.csv not present in this checkout")
class TestBestRecurrentSelection(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(RUNS_CSV)

    def test_all_five_recurrent_experiments_present(self):
        experiments = set(self.df["experiment"])
        for exp in ["M0", "M1", "M2", "M3", "M4"]:
            self.assertIn(exp, experiments)

    def test_m0_has_exactly_three_seeds_m1_to_m4_have_one(self):
        for seed in [42, 123, 456]:
            self.assertTrue(((self.df["experiment"] == "M0") & (self.df["seed"] == seed)).any())
        for exp in ["M1", "M2", "M3", "M4"]:
            sub = self.df[self.df["experiment"] == exp]
            self.assertEqual(len(sub), 1, f"{exp} must be single-seed")
            self.assertEqual(sub.iloc[0]["seed"], 42)

    def test_m4_is_the_highest_macro_f1_among_m0_to_m4(self):
        m0_mean = seed_statistics(self.df[self.df["experiment"] == "M0"]["macro_f1"].values)["mean"]
        candidates = {"M0": m0_mean}
        for exp in ["M1", "M2", "M3", "M4"]:
            candidates[exp] = self.df[self.df["experiment"] == exp]["macro_f1"].iloc[0]
        best = max(candidates, key=candidates.get)
        self.assertEqual(best, "M4")

    def test_m4_delta_vs_m0_exceeds_stability_spread(self):
        m0_f1s = self.df[self.df["experiment"] == "M0"]["macro_f1"].values
        m0_stats = seed_statistics(m0_f1s)
        m4_f1 = self.df[self.df["experiment"] == "M4"]["macro_f1"].iloc[0]
        delta = calculate_deltas(current_f1=m4_f1, baseline_m0_f1=m0_stats["mean"])["delta_vs_m0"]
        spread = float(m0_f1s.max() - m0_f1s.min())
        self.assertGreater(abs(delta), spread)


@unittest.skipUnless(BEST_RECURRENT_JSON.exists(), "results/best_recurrent.json not present in this checkout")
class TestBestRecurrentArtifact(unittest.TestCase):
    def setUp(self):
        self.summary = json.loads(BEST_RECURRENT_JSON.read_text(encoding="utf-8"))
        self.df = pd.read_csv(RUNS_CSV)

    def test_required_fields_present(self):
        for field in [
            "best_recurrent_experiment", "best_recurrent_macro_f1", "best_recurrent_delta_vs_m0",
            "tfidf_reference_macro_f1", "m0_mean_macro_f1", "m0_std_macro_f1",
        ]:
            self.assertIn(field, self.summary)

    def test_best_experiment_is_m4(self):
        self.assertEqual(self.summary["best_recurrent_experiment"], "M4")

    def test_best_macro_f1_matches_runs_csv(self):
        m4_row = self.df[(self.df["experiment"] == "M4") & (self.df["seed"] == 42)].iloc[0]
        self.assertAlmostEqual(self.summary["best_recurrent_macro_f1"], m4_row["macro_f1"], places=12)

    def test_m0_mean_and_std_match_independent_recompute(self):
        m0_f1s = self.df[self.df["experiment"] == "M0"]["macro_f1"].values
        stats = seed_statistics(m0_f1s)
        self.assertAlmostEqual(self.summary["m0_mean_macro_f1"], stats["mean"], places=12)
        self.assertAlmostEqual(self.summary["m0_std_macro_f1"], stats["std"], places=12)

    def test_delta_vs_m0_matches_calculate_deltas(self):
        expected = calculate_deltas(
            current_f1=self.summary["best_recurrent_macro_f1"],
            baseline_m0_f1=self.summary["m0_mean_macro_f1"],
        )["delta_vs_m0"]
        self.assertAlmostEqual(self.summary["best_recurrent_delta_vs_m0"], expected, places=12)

    def test_tfidf_reference_matches_its_own_result_file(self):
        tfidf_path = Path("results/tfidf_reference.json")
        if not tfidf_path.exists():
            self.skipTest("results/tfidf_reference.json not present")
        tfidf = json.loads(tfidf_path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(self.summary["tfidf_reference_macro_f1"], tfidf["macro_f1"], places=12)


if __name__ == "__main__":
    unittest.main()
