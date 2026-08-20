"""Text-preprocessing orchestration: frozen data -> model-ready inputs.

Ties together `src.data` (loading/labels), `src.text` (normalization primitive),
`src.keras_tokenizer` (M0-M4 vocabulary/sequences), and `src.embeddings` (GloVe for
M3) against the split already frozen in Task 2. Nothing here refits or alters the
split - it is loaded and verified, never regenerated.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.data import CLASS_TO_ID, TEXT_COL, deduplicate, load_raw
from src.fingerprint import verify_dataset_fingerprint, verify_split_fingerprint
from src.keras_tokenizer import KerasTokenizer, pad_sequences
from src.split import load_splits
from src.text import strip_bytes_wrapper

DATASET_MANIFEST_PATH = Path("data/dataset_manifest.json")
SPLIT_MANIFEST_PATH = Path("data/splits/split_manifest.json")
SPLIT_DIR = Path("data/splits")


def normalize_text(text: str) -> str:
    """The only normalization step the audit justified: undo the `b'...'` byte-repr
    wrapper found in 0.66% of narratives (Stage 1 finding). Casing, punctuation,
    digits, `XXXX`, and whitespace are all left untouched - none of them were found
    to be noise, and lowercasing/punctuation-stripping is the tokenizer's job, not
    a preprocessing step applied ahead of it."""
    return strip_bytes_wrapper(text)


def load_frozen_dataset(
    base_dir: str = ".",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the deduplicated dataset and the frozen split, verifying both against
    the manifests committed in Task 1/2. Raises loudly on any mismatch - this
    function never regenerates anything.

    `base_dir` lets callers outside the repo root (e.g. a notebook running from
    `notebooks/`) point at the same frozen files without duplicating path logic -
    pass `".."` from there, matching the convention already used in
    `notebooks/01_eda.ipynb`.
    """
    base = Path(base_dir)
    df_raw = load_raw(base / "data" / "combined_complaints.parquet")
    verify_dataset_fingerprint(df_raw, base / DATASET_MANIFEST_PATH)

    df = deduplicate(df_raw)
    train_idx, val_idx, test_idx = load_splits(base / SPLIT_DIR)
    verify_split_fingerprint(df, train_idx, val_idx, test_idx, base / SPLIT_MANIFEST_PATH)

    return df, train_idx, val_idx, test_idx


def fit_tokenizer_on_train(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    max_features: int,
    oov_token: str,
) -> KerasTokenizer:
    """Fits on `df.iloc[train_idx]` narratives only. This is the single fit call
    site in the project - `tests/test_preprocessing.py::test_fit_is_train_only`
    checks that refitting on validation/test text does not change `word_index`,
    which only holds if callers never do that in the first place."""
    tokenizer = KerasTokenizer(num_words=max_features, oov_token=oov_token)
    train_texts = df.iloc[train_idx][TEXT_COL].map(normalize_text)
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def build_sequences(
    texts: pd.Series,
    tokenizer: KerasTokenizer,
    max_len: int,
    padding: str = "pre",
    truncating: str = "post",
) -> np.ndarray:
    """normalized text -> integer sequences -> padded array. Row count in must
    equal row count out; this is asserted, not assumed."""
    normalized = texts.map(normalize_text)
    sequences = tokenizer.texts_to_sequences(normalized)
    if len(sequences) != len(texts):
        raise AssertionError(f"sequence count {len(sequences)} != input row count {len(texts)}")
    return pad_sequences(sequences, max_len=max_len, padding=padding, truncating=truncating)


def class_ids(labels: pd.Series) -> np.ndarray:
    """Label strings -> canonical integer ids (src.data.CLASS_TO_ID), the same
    mapping `src.evaluation.compute_metrics` assumes for integer-typed inputs. The
    single place label order gets turned into numbers, so every split and every
    model uses the same mapping.

    Fails loudly on an unrecognized label - `pandas.Series.map` would otherwise
    silently produce NaN for it, which then casts to a garbage int64 value instead
    of raising. That silence is exactly what "do not invent labels" rules out.
    """
    unknown = set(labels) - set(CLASS_TO_ID)
    if unknown:
        raise ValueError(f"Unrecognized label(s), not in src.data.LABELS: {sorted(unknown)}")
    return labels.map(CLASS_TO_ID).to_numpy(dtype=np.int64)


def oov_rate(texts: pd.Series, tokenizer: KerasTokenizer) -> dict[str, float]:
    """Share of tokens that fall outside the retained vocabulary (mapped to the OOV
    index or, without one, dropped). Computed at the token level, not the document
    level - a document with one rare word is not "OOV", most of it is fine."""
    oov_index = tokenizer.word_index.get(tokenizer.oov_token) if tokenizer.oov_token else None
    total = 0
    oov = 0
    for text in texts.map(normalize_text):
        for word in tokenizer.split_text(text):
            total += 1
            idx = tokenizer.word_index.get(word)
            is_oov = idx is None or (tokenizer.num_words is not None and idx >= tokenizer.num_words)
            if is_oov:
                oov += 1
    return {"tokens": total, "oov_tokens": oov, "oov_rate_%": round(100 * oov / total, 3) if total else 0.0}


def tokenize_for_distilbert(
    texts: pd.Series,
    max_len: int,
    checkpoint: str = "distilbert-base-uncased",
) -> dict[str, Any]:
    """Uses the DistilBERT tokenizer as-is - pretrained, never fit on this
    dataset. Separate code path from the Keras tokenizer above; the two must not
    be conflated."""
    tok = AutoTokenizer.from_pretrained(checkpoint)
    normalized = texts.map(normalize_text).tolist()
    encoded = tok(
        normalized,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "tokenizer": tok,
    }
