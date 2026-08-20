"""Unit tests for the TF-IDF + Logistic Regression auxiliary reference.

Tiny synthetic text throughout - this checks the pipeline's correctness (train-only
fitting, sparse output, label mapping, metric integration, namespace isolation
from the official ladder), not the real reference numbers, which come from
notebooks/04_tfidf_reference.ipynb against the frozen split.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from src.checkpoint import VALID_EXPERIMENTS
from src.data import CLASS_TO_ID, LABELS
from src.evaluation import compute_metrics
from src.tfidf_reference import (
    TfidfReferenceResult,
    build_result,
    fit_tfidf_reference,
    top_features_by_class,
    transform,
)


def synthetic_split():
    # Distinct, repeated vocabulary per class so a trivial linear model can
    # actually separate them - this is a pipeline-correctness check, not a
    # measurement of real-world lexical separability.
    vocab = {
        LABELS[0]: "checking savings account overdraft fee bank",
        LABELS[1]: "credit card charge dispute interest rate",
        LABELS[2]: "debt collector collection agency validate",
        LABELS[3]: "wire transfer zelle venmo crypto exchange",
        LABELS[4]: "student loan navient forbearance servicer",
    }
    rows = []
    for label, words in vocab.items():
        for i in range(20):
            rows.append({"text": f"{words} case number {i}", "label": label})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    n = len(df)
    train = df.iloc[: int(n * 0.7)]
    val = df.iloc[int(n * 0.7) : int(n * 0.85)]
    test = df.iloc[int(n * 0.85):]
    return train, val, test


class TestFitTransform(unittest.TestCase):
    def setUp(self):
        self.train, self.val, self.test = synthetic_split()
        self.vectorizer, self.model = fit_tfidf_reference(self.train["text"], self.train["label"])

    def test_fits_on_train(self):
        self.assertGreater(len(self.vectorizer.vocabulary_), 0)

    def test_transform_works_on_validation(self):
        X_val = transform(self.vectorizer, self.val["text"])
        self.assertEqual(X_val.shape[0], len(self.val))

    def test_transform_works_on_test(self):
        X_test = transform(self.vectorizer, self.test["text"])
        self.assertEqual(X_test.shape[0], len(self.test))

    def test_validation_and_test_do_not_alter_fitted_vocabulary(self):
        vocab_before = dict(self.vectorizer.vocabulary_)
        transform(self.vectorizer, self.val["text"])
        transform(self.vectorizer, self.test["text"])
        self.assertEqual(self.vectorizer.vocabulary_, vocab_before)

    def test_unseen_word_in_val_does_not_enter_vocabulary(self):
        novel = pd.Series(["a totally unprecedented zzzqqxx term never in train"])
        transform(self.vectorizer, novel)
        self.assertNotIn("zzzqqxx", self.vectorizer.vocabulary_)

    def test_output_matrix_is_sparse(self):
        X_test = transform(self.vectorizer, self.test["text"])
        self.assertTrue(sparse.issparse(X_test))

    def test_class_mapping_matches_canonical_order(self):
        # LogisticRegression's classes_ must be the same integer ids CLASS_TO_ID uses.
        np.testing.assert_array_equal(self.model.classes_, sorted(CLASS_TO_ID.values()))


class TestMetricsIntegration(unittest.TestCase):
    def setUp(self):
        self.train, self.val, self.test = synthetic_split()
        self.vectorizer, self.model = fit_tfidf_reference(self.train["text"], self.train["label"])

    def test_metrics_match_src_evaluation_directly(self):
        X_test = transform(self.vectorizer, self.test["text"])
        y_test = self.test["label"].map(CLASS_TO_ID).to_numpy()
        y_pred = self.model.predict(X_test)

        via_module = compute_metrics(y_test, y_pred, labels=LABELS)
        # build_result must produce numbers identical to calling compute_metrics
        # directly - no second metric implementation.
        result = build_result(
            self.vectorizer, self.model, X_test, y_test,
            val_macro_f1=0.0, dataset_content_sha256="x", preprocessing_version="pp-v1",
            runtime_seconds=0.1, git_commit="abc123",
        )
        self.assertAlmostEqual(result.macro_f1, via_module["macro_f1"], places=10)
        self.assertAlmostEqual(result.accuracy, via_module["accuracy"], places=10)
        self.assertEqual(result.confusion_matrix, via_module["confusion_matrix"])

    def test_reasonable_separation_on_synthetic_data(self):
        # Not a real-world accuracy claim - just confirms the pipeline actually
        # learns something on cleanly separable synthetic classes.
        X_test = transform(self.vectorizer, self.test["text"])
        y_test = self.test["label"].map(CLASS_TO_ID).to_numpy()
        y_pred = self.model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, labels=LABELS)
        self.assertGreater(metrics["macro_f1"], 0.5)


class TestResultRecord(unittest.TestCase):
    def setUp(self):
        self.train, self.val, self.test = synthetic_split()
        self.vectorizer, self.model = fit_tfidf_reference(self.train["text"], self.train["label"])
        X_test = transform(self.vectorizer, self.test["text"])
        y_test = self.test["label"].map(CLASS_TO_ID).to_numpy()
        self.result = build_result(
            self.vectorizer, self.model, X_test, y_test,
            val_macro_f1=0.42, dataset_content_sha256="deadbeef", preprocessing_version="pp-v1",
            runtime_seconds=1.23, git_commit="abc123",
        )

    def test_labeled_as_auxiliary_reference_not_an_official_id(self):
        self.assertEqual(self.result.label, "TFIDF_REFERENCE")
        self.assertNotIn(self.result.label, VALID_EXPERIMENTS)

    def test_official_experiment_ids_untouched(self):
        self.assertEqual(VALID_EXPERIMENTS, ["M0", "M1", "M2", "M3", "M4", "D0"])

    def test_save_and_load_round_trip(self):
        tmp = Path(tempfile.mkdtemp()) / "tfidf_reference.json"
        self.result.save(tmp)
        loaded = TfidfReferenceResult.load(tmp)
        self.assertEqual(loaded.macro_f1, self.result.macro_f1)
        self.assertEqual(loaded.label, "TFIDF_REFERENCE")

    def test_record_carries_full_traceability(self):
        d = self.result.to_dict()
        for field in ("dataset_content_sha256", "preprocessing_version", "git_commit",
                      "sklearn_version", "python_version", "tfidf_config", "logreg_config"):
            self.assertIn(field, d)
            self.assertTrue(d[field])


class TestTopFeatures(unittest.TestCase):
    def test_returns_k_terms_per_class(self):
        train, _, _ = synthetic_split()
        vectorizer, model = fit_tfidf_reference(train["text"], train["label"])
        top = top_features_by_class(vectorizer, model, k=5)
        self.assertEqual(set(top.keys()), set(LABELS))
        for terms in top.values():
            self.assertLessEqual(len(terms), 5)
            self.assertGreater(len(terms), 0)


if __name__ == "__main__":
    unittest.main()
