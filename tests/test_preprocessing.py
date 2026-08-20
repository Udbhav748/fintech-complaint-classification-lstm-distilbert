"""Unit tests for the preprocessing orchestration layer.

Small synthetic examples for the pure logic; a couple of tests exercise the real
frozen project split (fast - hashing + loading only) to prove the pipeline actually
runs against what Task 1/2 froze, matching the precedent in tests/test_split.py.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data import CLASS_TO_ID, LABELS, LABEL_COL, TEXT_COL
from src.embeddings import build_embedding_matrix
from src.keras_tokenizer import KerasTokenizer
from src.preprocessing import (
    build_sequences,
    class_ids,
    fit_tokenizer_on_train,
    load_frozen_dataset,
    normalize_text,
    oov_rate,
    tokenize_for_distilbert,
)


class TestNormalize(unittest.TestCase):
    def test_strips_bytes_wrapper(self):
        self.assertEqual(normalize_text("b'hello world'"), "hello world")

    def test_leaves_casing_punctuation_digits_and_redactions_untouched(self):
        text = "URGENT! Chime charged me $150.00 on XX/XX/2024, account XXXX."
        self.assertEqual(normalize_text(text), text)

    def test_deterministic(self):
        text = "b'repeat me'"
        self.assertEqual(normalize_text(text), normalize_text(text))


class TestClassIds(unittest.TestCase):
    def test_matches_canonical_label_order(self):
        ids = class_ids(pd.Series(LABELS))
        np.testing.assert_array_equal(ids, np.arange(len(LABELS)))

    def test_unknown_label_fails_loudly(self):
        with self.assertRaises(ValueError):
            class_ids(pd.Series(["not a real product"]))


class TestFitTokenizerOnTrainOnly(unittest.TestCase):
    def test_val_and_test_only_words_absent_from_vocabulary(self):
        df = pd.DataFrame(
            {
                TEXT_COL: [
                    "train word alpha", "train word beta",  # train
                    "validation word gamma",                 # val
                    "test word delta",                       # test
                ],
                LABEL_COL: [LABELS[0]] * 4,
            }
        )
        train_idx = np.array([0, 1])
        tok = fit_tokenizer_on_train(df, train_idx, max_features=100, oov_token="<OOV>")
        self.assertIn("train", tok.word_index)
        self.assertIn("alpha", tok.word_index)
        self.assertNotIn("gamma", tok.word_index)
        self.assertNotIn("delta", tok.word_index)


class TestBuildSequences(unittest.TestCase):
    def setUp(self):
        self.tok = KerasTokenizer(num_words=50, oov_token="<OOV>")
        self.tok.fit_on_texts(["one two three", "four five"])

    def test_row_count_preserved(self):
        texts = pd.Series(["one two", "three four", "unseen words here"])
        out = build_sequences(texts, self.tok, max_len=8)
        self.assertEqual(out.shape[0], len(texts))

    def test_shape_matches_max_len(self):
        texts = pd.Series(["one two three"])
        self.assertEqual(build_sequences(texts, self.tok, max_len=128).shape, (1, 128))
        self.assertEqual(build_sequences(texts, self.tok, max_len=256).shape, (1, 256))

    def test_deterministic(self):
        texts = pd.Series(["one two three", "four five"])
        a = build_sequences(texts, self.tok, max_len=16)
        b = build_sequences(texts, self.tok, max_len=16)
        np.testing.assert_array_equal(a, b)


class TestOovRate(unittest.TestCase):
    def test_all_known_words_zero_oov(self):
        tok = KerasTokenizer(oov_token="<OOV>").fit_on_texts(["a b c"])
        stats = oov_rate(pd.Series(["a b c"]), tok)
        self.assertEqual(stats["oov_rate_%"], 0.0)

    def test_all_unknown_words_full_oov(self):
        tok = KerasTokenizer(oov_token="<OOV>").fit_on_texts(["a b c"])
        stats = oov_rate(pd.Series(["zzz yyy xxx"]), tok)
        self.assertEqual(stats["oov_rate_%"], 100.0)


class TestGloveEmbeddingMatrix(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.glove_path = Path(self.tmpdir.name) / "fake_glove.txt"
        vec = " ".join(f"{x:.4f}" for x in np.linspace(0.1, 0.4, 4))
        self.glove_path.write_text(f"alpha {vec}\nbeta {vec}\n", encoding="utf-8")
        self.word_index = {"<OOV>": 1, "alpha": 2, "beta": 3, "gamma": 4}

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_shape_and_dimension(self):
        matrix, meta = build_embedding_matrix(self.word_index, vocab_size=5, glove_path=self.glove_path, dim=4)
        self.assertEqual(matrix.shape, (5, 4))
        self.assertEqual(meta["dim"], 4)

    def test_padding_row_is_zero(self):
        matrix, _ = build_embedding_matrix(self.word_index, vocab_size=5, glove_path=self.glove_path, dim=4)
        np.testing.assert_array_equal(matrix[0], np.zeros(4))

    def test_matched_word_gets_glove_vector(self):
        matrix, meta = build_embedding_matrix(self.word_index, vocab_size=5, glove_path=self.glove_path, dim=4)
        expected = np.linspace(0.1, 0.4, 4).astype(np.float32)
        np.testing.assert_allclose(matrix[self.word_index["alpha"]], expected, atol=1e-3)
        self.assertEqual(meta["matched_types"], 2)  # alpha, beta

    def test_unmatched_words_get_independent_vectors(self):
        # oov(1) and gamma(4) both miss GloVe; must not collapse to the same vector.
        matrix, _ = build_embedding_matrix(self.word_index, vocab_size=5, glove_path=self.glove_path, dim=4)
        self.assertFalse(np.array_equal(matrix[1], matrix[4]))

    def test_deterministic_given_seed(self):
        a, _ = build_embedding_matrix(self.word_index, vocab_size=5, glove_path=self.glove_path, dim=4, seed=7)
        b, _ = build_embedding_matrix(self.word_index, vocab_size=5, glove_path=self.glove_path, dim=4, seed=7)
        np.testing.assert_array_equal(a, b)


class TestDistilBertTokenization(unittest.TestCase):
    def test_shapes_and_special_tokens(self):
        texts = pd.Series(["short complaint text", "another one here"])
        out = tokenize_for_distilbert(texts, max_len=16)
        tok = out["tokenizer"]
        self.assertEqual(out["input_ids"].shape, (2, 16))
        self.assertEqual(out["attention_mask"].shape, (2, 16))
        self.assertTrue((out["input_ids"][:, 0] == tok.cls_token_id).all())

    def test_truncation_at_max_len(self):
        long_text = pd.Series([" ".join(["word"] * 500)])
        out = tokenize_for_distilbert(long_text, max_len=32)
        self.assertEqual(out["input_ids"].shape, (1, 32))


class TestFrozenDatasetIntegration(unittest.TestCase):
    """Exercises the real project artifacts frozen in Task 1/2 - fast (hash +
    load only), matching the precedent already set in tests/test_split.py."""

    def test_loads_and_verifies_against_committed_manifests(self):
        df, train_idx, val_idx, test_idx = load_frozen_dataset()
        self.assertEqual(len(df), 101802)
        self.assertEqual(len(train_idx), 81442)
        self.assertEqual(len(val_idx), 10179)
        self.assertEqual(len(test_idx), 10181)
        # No cross-split overlap, no silent row loss.
        self.assertEqual(len(set(train_idx) & set(val_idx)), 0)
        self.assertEqual(len(set(train_idx) & set(test_idx)), 0)
        self.assertEqual(len(train_idx) + len(val_idx) + len(test_idx), len(df))


if __name__ == "__main__":
    unittest.main()
