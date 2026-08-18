"""LSTM vocabulary builder and tokenization preparation for Phase 4.

Builds and fits a word-level vocabulary strictly on the TRAINING split.
Evaluates OOV rates and token coverage across train, validation, and test sets.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Standard word regex: lowercased alphanumeric tokens, keeping contractions like don't, can't
WORD_REGEX = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_INDEX = 0
UNK_INDEX = 1


def tokenize_text(text: str) -> list[str]:
    """Tokenize lowercased text into word tokens preserving words and contractions."""
    if not isinstance(text, str):
        return []
    return WORD_REGEX.findall(text.lower())


@dataclass
class Vocabulary:
    """Word-level vocabulary with index mapping and OOV handling."""

    word2idx: dict[str, int]
    idx2word: dict[int, str]
    word_counts: dict[str, int]
    pad_token: str = PAD_TOKEN
    unk_token: str = UNK_TOKEN
    pad_idx: int = PAD_INDEX
    unk_idx: int = UNK_INDEX

    @property
    def size(self) -> int:
        return len(self.word2idx)

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a list of word tokens to token indices."""
        return [self.word2idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        """Convert a list of token indices back to word tokens."""
        return [self.idx2word.get(idx, self.unk_token) for idx in indices]

    def to_dict(self) -> dict[str, Any]:
        """Serialize vocabulary to JSON-serializable dictionary."""
        return {
            "size": self.size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "pad_idx": self.pad_idx,
            "unk_idx": self.unk_idx,
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "word_counts": self.word_counts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Vocabulary:
        """Load vocabulary from dictionary."""
        word2idx = d["word2idx"]
        idx2word = {int(k): v for k, v in d["idx2word"].items()}
        word_counts = d.get("word_counts", {})
        return cls(
            word2idx=word2idx,
            idx2word=idx2word,
            word_counts=word_counts,
            pad_token=d.get("pad_token", PAD_TOKEN),
            unk_token=d.get("unk_token", UNK_TOKEN),
            pad_idx=d.get("pad_idx", PAD_INDEX),
            unk_idx=d.get("unk_idx", UNK_INDEX),
        )


def build_lstm_vocabulary(
    train_texts: list[str],
    max_vocab_size: int = 25000,
    min_freq: int = 2,
) -> Vocabulary:
    """Build a word vocabulary strictly from training texts.

    Args:
        train_texts: List of narrative texts from the TRAINING split only.
        max_vocab_size: Maximum vocabulary size including special tokens.
        min_freq: Minimum occurrence count for a word to be included.

    Returns:
        Vocabulary object fitted strictly on training data.
    """
    logger.info("Building LSTM vocabulary from %d training texts...", len(train_texts))
    counter = Counter()

    for text in train_texts:
        tokens = tokenize_text(text)
        counter.update(tokens)

    logger.info("Total unique word tokens in training corpus: %d", len(counter))

    # Reserve special tokens
    word2idx = {PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX}
    idx2word = {PAD_INDEX: PAD_TOKEN, UNK_INDEX: UNK_TOKEN}

    current_idx = 2
    # Filter by min_freq and take top words up to max_vocab_size
    most_common = counter.most_common()
    for word, count in most_common:
        if count < min_freq:
            break
        if len(word2idx) >= max_vocab_size:
            break
        if word not in word2idx:
            word2idx[word] = current_idx
            idx2word[current_idx] = word
            current_idx += 1

    word_counts_dict = {w: counter[w] for w in word2idx if w not in (PAD_TOKEN, UNK_TOKEN)}

    vocab = Vocabulary(
        word2idx=word2idx,
        idx2word=idx2word,
        word_counts=word_counts_dict,
    )
    logger.info(
        "Fitted LSTM vocabulary: %d words (max_size=%d, min_freq=%d)",
        vocab.size,
        max_vocab_size,
        min_freq,
    )
    return vocab


def evaluate_vocabulary_coverage(
    vocab: Vocabulary,
    texts: list[str],
    split_name: str = "split",
) -> dict[str, Any]:
    """Evaluate token coverage and OOV rate of a vocabulary on a set of texts."""
    total_tokens = 0
    oov_tokens = 0
    unique_tokens = set()
    unique_oov_tokens = set()

    for text in texts:
        tokens = tokenize_text(text)
        total_tokens += len(tokens)
        for t in tokens:
            unique_tokens.add(t)
            if t not in vocab.word2idx:
                oov_tokens += 1
                unique_oov_tokens.add(t)

    token_coverage_pct = ((total_tokens - oov_tokens) / total_tokens * 100.0) if total_tokens > 0 else 0.0
    oov_rate_pct = (oov_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0
    unique_coverage_pct = (
        ((len(unique_tokens) - len(unique_oov_tokens)) / len(unique_tokens) * 100.0) if unique_tokens else 0.0
    )

    return {
        "split_name": split_name,
        "total_tokens": total_tokens,
        "oov_tokens": oov_tokens,
        "token_coverage_percentage": round(token_coverage_pct, 2),
        "oov_rate_percentage": round(oov_rate_pct, 2),
        "unique_tokens_count": len(unique_tokens),
        "unique_oov_count": len(unique_oov_tokens),
        "unique_word_coverage_percentage": round(unique_coverage_pct, 2),
    }
