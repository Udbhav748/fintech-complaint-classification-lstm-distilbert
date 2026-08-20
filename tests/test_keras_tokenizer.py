"""Unit tests for the TensorFlow-free Keras-compatible tokenizer.

Behavioral equivalence with real tf.keras.preprocessing.text.Tokenizer is checked
separately in scripts/verify_keras_tokenizer_equivalence.py (requires TensorFlow,
not run here). These tests check the algorithm this project actually relies on:
hand-computed word_index, OOV handling, padding shape/side, and determinism.
"""

import unittest

import numpy as np

from src.keras_tokenizer import KerasTokenizer, pad_sequences


class TestKerasTokenizerFit(unittest.TestCase):
    def test_word_index_matches_hand_computed_frequency_order(self):
        texts = ["a a a b b c", "a b"]
        # counts: a=4, b=3, c=1 -> descending frequency order
        tok = KerasTokenizer(oov_token=None)
        tok.fit_on_texts(texts)
        self.assertEqual(tok.word_index, {"a": 1, "b": 2, "c": 3})

    def test_ties_broken_by_first_occurrence_order(self):
        texts = ["zebra apple mango"]  # all count=1, first-seen order z, a, m
        tok = KerasTokenizer(oov_token=None)
        tok.fit_on_texts(texts)
        self.assertEqual(tok.word_index, {"zebra": 1, "apple": 2, "mango": 3})

    def test_oov_token_gets_index_1_and_shifts_others(self):
        texts = ["a a b"]
        tok = KerasTokenizer(oov_token="<OOV>")
        tok.fit_on_texts(texts)
        self.assertEqual(tok.word_index["<OOV>"], 1)
        self.assertEqual(tok.word_index["a"], 2)
        self.assertEqual(tok.word_index["b"], 3)

    def test_index_zero_never_assigned(self):
        tok = KerasTokenizer(oov_token="<OOV>")
        tok.fit_on_texts(["a b c"])
        self.assertNotIn(0, tok.word_index.values())


class TestTextsToSequences(unittest.TestCase):
    def test_known_words_map_to_fitted_indices(self):
        tok = KerasTokenizer(oov_token=None)
        tok.fit_on_texts(["a b c"])
        self.assertEqual(tok.texts_to_sequences(["a c"]), [[tok.word_index["a"], tok.word_index["c"]]])

    def test_unseen_word_maps_to_oov_index(self):
        tok = KerasTokenizer(oov_token="<OOV>")
        tok.fit_on_texts(["a b c"])
        oov_idx = tok.word_index["<OOV>"]
        self.assertEqual(tok.texts_to_sequences(["a zzz"]), [[tok.word_index["a"], oov_idx]])

    def test_unseen_word_dropped_without_oov_token(self):
        tok = KerasTokenizer(oov_token=None)
        tok.fit_on_texts(["a b c"])
        self.assertEqual(tok.texts_to_sequences(["a zzz"]), [[tok.word_index["a"]]])

    def test_num_words_caps_vocabulary_at_texts_to_sequences_time(self):
        # counts: a=5, b=4, c=3, d=1 -> indices (no oov) a=1 b=2 c=3 d=4
        tok = KerasTokenizer(num_words=3, oov_token="<OOV>")
        tok.fit_on_texts(["a a a a a b b b b c c c d"])
        # with oov: oov=1, a=2, b=3, c=4, d=5; num_words=3 keeps indices < 3 -> only oov(1), a(2)
        seq = tok.texts_to_sequences(["a b c d"])[0]
        oov_idx = tok.word_index["<OOV>"]
        self.assertEqual(seq, [tok.word_index["a"], oov_idx, oov_idx, oov_idx])

    def test_fitting_again_overwrites_not_merges(self):
        """Guards the leakage rule at the tokenizer level: `src.preprocessing` must
        call fit_on_texts exactly once, on train. This test documents that calling
        it a second time replaces word_index rather than extending it, so an
        accidental second fit call is immediately visible in its effect."""
        tok = KerasTokenizer(oov_token=None)
        tok.fit_on_texts(["a b"])
        first = dict(tok.word_index)
        tok.fit_on_texts(["totally different words here"])
        self.assertNotEqual(tok.word_index, first)
        self.assertNotIn("a", tok.word_index)


class TestPadSequences(unittest.TestCase):
    def test_shape(self):
        out = pad_sequences([[1, 2], [3]], max_len=5)
        self.assertEqual(out.shape, (2, 5))

    def test_pre_padding_places_zeros_first(self):
        out = pad_sequences([[1, 2]], max_len=4, padding="pre")
        np.testing.assert_array_equal(out[0], [0, 0, 1, 2])

    def test_post_padding_places_zeros_last(self):
        out = pad_sequences([[1, 2]], max_len=4, padding="post")
        np.testing.assert_array_equal(out[0], [1, 2, 0, 0])

    def test_post_truncating_keeps_the_start(self):
        out = pad_sequences([[1, 2, 3, 4, 5]], max_len=3, truncating="post")
        np.testing.assert_array_equal(out[0], [1, 2, 3])

    def test_pre_truncating_keeps_the_end(self):
        out = pad_sequences([[1, 2, 3, 4, 5]], max_len=3, truncating="pre")
        np.testing.assert_array_equal(out[0], [3, 4, 5])

    def test_max_len_128_and_256_shapes(self):
        seqs = [[1] * 300]
        self.assertEqual(pad_sequences(seqs, max_len=128).shape, (1, 128))
        self.assertEqual(pad_sequences(seqs, max_len=256).shape, (1, 256))

    def test_empty_sequence_is_all_padding(self):
        out = pad_sequences([[]], max_len=4, padding="pre")
        np.testing.assert_array_equal(out[0], [0, 0, 0, 0])


class TestDeterminism(unittest.TestCase):
    def test_fit_twice_on_same_texts_is_identical(self):
        texts = ["the quick brown fox", "the lazy dog", "fox and dog"]
        a = KerasTokenizer(num_words=100, oov_token="<OOV>").fit_on_texts(texts)
        b = KerasTokenizer(num_words=100, oov_token="<OOV>").fit_on_texts(texts)
        self.assertEqual(a.word_index, b.word_index)
        self.assertEqual(a.texts_to_sequences(texts), b.texts_to_sequences(texts))


class TestSerialization(unittest.TestCase):
    def test_roundtrip_preserves_word_index(self):
        import json

        tok = KerasTokenizer(num_words=50, oov_token="<OOV>").fit_on_texts(["a b c a a"])
        restored = KerasTokenizer.from_dict(json.loads(tok.to_json()))
        self.assertEqual(tok.word_index, restored.word_index)
        self.assertEqual(tok.num_words, restored.num_words)
        self.assertEqual(tok.texts_to_sequences(["a c"]), restored.texts_to_sequences(["a c"]))


if __name__ == "__main__":
    unittest.main()
