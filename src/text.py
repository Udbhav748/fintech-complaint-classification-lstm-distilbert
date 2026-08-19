"""Tokenisation, length and text-artifact measurement."""

import re

import numpy as np
import pandas as pd

# Default `filters` of tf.keras.preprocessing.text.Tokenizer. Replicated here so
# the audit can measure Keras sequence lengths without a TensorFlow install; the
# apostrophe is deliberately not a filter character, which matters for GloVe
# lookup (see notebooks/01_eda.ipynb, "GloVe coverage").
KERAS_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
_KERAS_TABLE = str.maketrans(KERAS_FILTERS, " " * len(KERAS_FILTERS))

# Redaction and formatting patterns specific to the CFPB public extract.
ARTIFACT_PATTERNS = {
    "XXXX redaction": r"\bX{2,}\b",
    "XX/XX date": r"XX/XX/",
    "year> placeholder": r"year>",
    "{$...} amount": r"\{\$[\d.,]+\}",
    "bytes repr wrapper": r"^b['\"]",
    "url": r"https?://|www\.",
    "email": r"\S+@\S+\.\w+",
    "html tag": r"<[a-z/][^>]*>",
    "non-ascii": r"[^\x00-\x7f]",
    "blank line": r"\n\s*\n",
}


def keras_word_sequence(text: str) -> list[str]:
    return [w for w in text.lower().translate(_KERAS_TABLE).split(" ") if w]


def wordpiece_lengths(texts, tokenizer, batch_size: int = 2000) -> np.ndarray:
    """Token counts including [CLS]/[SEP], untruncated."""
    texts = list(texts)
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i : i + batch_size], add_special_tokens=True, truncation=False)
        out.extend(len(ids) for ids in enc["input_ids"])
    return np.array(out, dtype=np.int32)


def length_summary(lengths, percentiles=(0.5, 0.75, 0.9, 0.95, 0.99)) -> pd.Series:
    s = pd.Series(lengths)
    out = {"min": s.min(), "mean": round(s.mean(), 1), "std": round(s.std(), 1)}
    out.update({f"p{int(p * 100)}": int(s.quantile(p)) for p in percentiles})
    out["max"] = s.max()
    return pd.Series(out)


def truncation_table(lengths, max_lens) -> pd.DataFrame:
    """Share of documents cut off at each max_len, and the share of all tokens
    that survives. The second column is the one that matters for M4: a high
    truncation rate is harmless if the tail carries little text."""
    lengths = np.asarray(lengths)
    rows = []
    for m in max_lens:
        rows.append(
            {
                "max_len": m,
                "truncated_%": round(100 * (lengths > m).mean(), 1),
                "tokens_kept_%": round(100 * np.minimum(lengths, m).sum() / lengths.sum(), 1),
            }
        )
    return pd.DataFrame(rows).set_index("max_len")


def artifact_counts(texts: pd.Series) -> pd.DataFrame:
    rows = []
    for name, pattern in ARTIFACT_PATTERNS.items():
        hits = texts.str.contains(pattern, regex=True, na=False)
        rows.append({"artifact": name, "docs": int(hits.sum()), "pct": round(100 * hits.mean(), 2)})
    return pd.DataFrame(rows).set_index("artifact")


def count_matches(texts: pd.Series, pattern: str) -> pd.Series:
    return texts.str.count(pattern)


def max_similarity(queries, index, chunk_size: int = 500, exclude=None) -> np.ndarray:
    """Highest cosine similarity of each query row against every index row.

    Both arguments are L2-normalised sparse TF-IDF matrices, so the dot product
    is the cosine. `exclude` gives, per query, an index row to mask out - used
    when the queries are a subset of the index and would otherwise match
    themselves.
    """
    best = np.empty(queries.shape[0], dtype=np.float32)
    for i in range(0, queries.shape[0], chunk_size):
        block = (queries[i : i + chunk_size] @ index.T).toarray()
        if exclude is not None:
            rows = np.arange(block.shape[0])
            block[rows, exclude[i : i + chunk_size]] = -1.0
        best[i : i + chunk_size] = block.max(axis=1)
    return best


def top_terms_by_log_odds(counts, terms, mask, k: int = 20) -> list[str]:
    """Terms most over-represented in one class relative to the rest."""
    total = np.asarray(counts.sum(0)).ravel().astype(float)
    inside = np.asarray(counts[mask].sum(0)).ravel().astype(float)
    outside = total - inside
    score = np.log((inside + 1) / (inside.sum() + 1)) - np.log((outside + 1) / (outside.sum() + 1))
    return list(np.asarray(terms)[np.argsort(-score)[:k]])


def strip_bytes_wrapper(text: str) -> str:
    """Undo the `b'...'` byte-string repr that leaked into 0.66% of narratives."""
    m = re.match(r"^b(['\"])(.*)\1$", text, flags=re.S)
    if not m:
        return text
    return m.group(2).replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
