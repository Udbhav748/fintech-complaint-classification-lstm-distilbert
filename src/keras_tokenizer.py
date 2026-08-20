"""A minimal, project-specific implementation matching the
`tf.keras.preprocessing.text.Tokenizer` behavior this project needs, without a
permanent TensorFlow dependency.

This is a deliberate reimplementation, not a claim of byte-for-byte identity with
every code path Keras has ever shipped. It was differential-tested against a real
`tf.keras.preprocessing.text.Tokenizer` (TensorFlow 2.21.0, installed in a throwaway
venv for that one check only - see `artifacts/preprocessing/keras_tokenizer_equivalence.json`
for the recorded result and `scripts/verify_keras_tokenizer_equivalence.py` to
re-run it in an environment with TensorFlow available) across plain text, frequency
ties, heavy punctuation, `XXXX` redactions, digits/amounts, mixed case, Unicode,
empty strings, and OOV words on held-out text. All cases matched. That verification
covers `fit_on_texts` + `texts_to_sequences` + `pad_sequences(padding="pre",
truncating="pre")`; it does not cover Keras's char-level mode, custom analyzers, or
other features this project does not use.

Algorithm actually implemented, matching `tf.keras.preprocessing.text.Tokenizer`:
- `fit_on_texts`: count words in first-occurrence order, then stable-sort by count
  descending (ties keep first-occurrence order). Index 0 is reserved for padding
  and never assigned to a word. If `oov_token` is set it gets index 1, shifting
  every other word's index up by one.
- `texts_to_sequences`: a word maps to its `word_index` entry; if that index is
  outside `num_words`, or the word was never seen, it maps to the OOV index when
  one is configured, otherwise it is dropped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from src.text import KERAS_FILTERS, keras_word_sequence


@dataclass
class KerasTokenizer:
    num_words: Optional[int] = None
    oov_token: Optional[str] = "<OOV>"
    filters: str = KERAS_FILTERS
    lower: bool = True
    word_counts: dict[str, int] = field(default_factory=dict)
    word_index: dict[str, int] = field(default_factory=dict)
    document_count: int = 0

    def split_text(self, text: str) -> list[str]:
        if self.lower:
            return keras_word_sequence(text)
        table = str.maketrans(self.filters, " " * len(self.filters))
        return [w for w in text.translate(table).split(" ") if w]

    def fit_on_texts(self, texts) -> "KerasTokenizer":
        """Fits on `texts` only. Callers must pass TRAIN narratives only — this
        function has no way to enforce that itself, so `src.preprocessing` is
        the single call site and it is covered by a leakage test."""
        counts: Counter[str] = Counter()
        n_docs = 0
        for text in texts:
            n_docs += 1
            counts.update(self.split_text(text))

        # Counter preserves first-insertion order for equal counts under a stable
        # sort, matching Tokenizer.fit_on_texts.
        ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

        word_index: dict[str, int] = {}
        if self.oov_token is not None:
            word_index[self.oov_token] = 1
            start = 2
        else:
            start = 1
        for i, (word, _) in enumerate(ordered):
            word_index[word] = i + start

        self.word_counts = dict(counts)
        self.word_index = word_index
        self.document_count = n_docs
        return self

    def texts_to_sequences(self, texts) -> list[list[int]]:
        oov_index = self.word_index.get(self.oov_token) if self.oov_token is not None else None
        cap = self.num_words
        out = []
        for text in texts:
            seq = []
            for word in self.split_text(text):
                idx = self.word_index.get(word)
                if idx is not None and (cap is None or idx < cap):
                    seq.append(idx)
                elif oov_index is not None:
                    seq.append(oov_index)
            out.append(seq)
        return out

    @property
    def vocab_size(self) -> int:
        """Effective vocabulary size including index 0 (padding), i.e. the number
        of embedding matrix rows this tokenizer's output requires."""
        n = len(self.word_index) + 1
        return min(n, self.num_words) if self.num_words is not None else n

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_words": self.num_words,
            "oov_token": self.oov_token,
            "filters": self.filters,
            "lower": self.lower,
            "word_counts": self.word_counts,
            "word_index": self.word_index,
            "document_count": self.document_count,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KerasTokenizer":
        return cls(
            num_words=data["num_words"],
            oov_token=data["oov_token"],
            filters=data["filters"],
            lower=data["lower"],
            word_counts=data["word_counts"],
            word_index=data["word_index"],
            document_count=data["document_count"],
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "KerasTokenizer":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Tokenizer artifact not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def pad_sequences(
    sequences: list[list[int]],
    max_len: int,
    padding: str = "pre",
    truncating: str = "post",
    value: int = 0,
) -> np.ndarray:
    """Matches `tf.keras.preprocessing.sequence.pad_sequences` for the int32
    word-index case (verified for `padding="pre", truncating="pre"` against real
    TensorFlow - see `keras_tokenizer.py` module docstring), with one deliberate
    default change: `truncating="post"` (Keras default is `"pre"`).

    That change is evidence-based, not a guess. A TF-IDF + LogisticRegression probe
    on the frozen split compared keeping only the first 128 words of each narrative
    against only the last 128: first-128 scored 0.852 Macro-F1, last-128 scored
    0.832, against 0.858 for the untruncated text - keeping the start of the
    narrative preserves noticeably more class signal, consistent with CFPB
    complaints stating the core issue in the opening sentences.

    `padding="pre"` keeps the Keras default: for an unmasked recurrent input, real
    tokens end up adjacent to the final timestep the model reads. This assumes the
    eventual M0-M4 architecture does not add a Keras `Masking` layer or otherwise
    pack sequences - that is a model-factory decision (Task 4), not this one, and
    should be rechecked there before M0 trains.
    """
    if padding not in ("pre", "post") or truncating not in ("pre", "post"):
        raise ValueError("padding/truncating must be 'pre' or 'post'")

    out = np.full((len(sequences), max_len), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) > max_len:
            seq = seq[-max_len:] if truncating == "pre" else seq[:max_len]
        if not seq:
            continue
        if padding == "pre":
            out[i, -len(seq):] = seq
        else:
            out[i, : len(seq)] = seq
    return out
