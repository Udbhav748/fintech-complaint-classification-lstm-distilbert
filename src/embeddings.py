"""GloVe vocabulary lookup, coverage measurement, and embedding matrix construction."""

from pathlib import Path

import numpy as np
import pandas as pd

GLOVE_PATH = Path("data/embeddings/glove.6B.100d.txt")
GLOVE_DIM = 100


def load_glove_vocab(path: Path = GLOVE_PATH) -> set[str]:
    """Only the words are needed to answer the coverage question; loading the
    400k x 100 vectors would cost ~160 MB for nothing at audit time."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - download glove.6B.100d and place it there")
    with open(path, encoding="utf-8") as f:
        return {line.split(" ", 1)[0] for line in f}


def coverage(counter, glove_vocab: set[str], vocab_sizes) -> pd.DataFrame:
    """Type and token coverage for the most frequent `n` dataset words.

    Type coverage answers "how many embedding rows stay random"; token coverage
    answers "how much of the text a model actually reads is pretrained". They
    diverge sharply on a long-tailed vocabulary, so both are reported.
    """
    corpus_tokens = sum(counter.values())
    rows = []
    for n in vocab_sizes:
        top = counter.most_common(n) if n else counter.most_common()
        hit_types = sum(1 for w, _ in top if w in glove_vocab)
        hit_tokens = sum(c for w, c in top if w in glove_vocab)
        kept_tokens = sum(c for _, c in top)
        rows.append(
            {
                "vocab_size": n or len(counter),
                "types_in_glove_%": round(100 * hit_types / len(top), 1),
                "tokens_in_glove_%": round(100 * hit_tokens / kept_tokens, 2),
                "corpus_covered_%": round(100 * hit_tokens / corpus_tokens, 2),
            }
        )
    return pd.DataFrame(rows).set_index("vocab_size")


def missing_terms(counter, glove_vocab: set[str], vocab_size: int, k: int = 30):
    top = counter.most_common(vocab_size)
    return [(w, c) for w, c in top if w not in glove_vocab][:k]


def build_embedding_matrix(
    word_index: dict[str, int],
    vocab_size: int,
    glove_path: Path = GLOVE_PATH,
    dim: int = GLOVE_DIM,
    seed: int = 42,
) -> tuple["np.ndarray", dict]:
    """Aligns GloVe vectors to a Keras tokenizer's `word_index`.

    Row 0 (padding) is zero. A word with a GloVe hit gets that vector. A word
    without one gets its own independently-drawn small random vector rather than
    zero or a shared placeholder - `rng.uniform(size=(vocab_size, dim))` draws
    `vocab_size * dim` independent values up front, so no two unmatched words start
    identical (tested in `tests/test_preprocessing.py`). Coverage is measured
    against `vocab_size` rows actually used by the model, not the full dataset
    vocabulary, since that is the number that matters for M3.

    Trainability is a property of the Keras `Embedding` layer, not of this matrix -
    `data_config.glove.trainable = true` means the *layer* updates every row
    (matched, unmatched, and padding alike) during training unless the model
    additionally masks index 0. This function only fixes deterministic starting
    values; whether the padding row's gradient is actually excluded is a model
    factory decision (Task 4), not this one.

    Streams the 347 MB GloVe file once rather than loading all 400k vectors, since
    only the rows this tokenizer can address are needed.
    """
    needed = {w for w, i in word_index.items() if i < vocab_size}
    found: dict[str, "np.ndarray"] = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            word, _, rest = line.partition(" ")
            if word in needed:
                found[word] = np.fromstring(rest, sep=" ", dtype=np.float32)
                if len(found) == len(needed):
                    break

    rng = np.random.default_rng(seed)
    matrix = rng.uniform(-0.05, 0.05, size=(vocab_size, dim)).astype(np.float32)
    matrix[0] = 0.0  # padding row

    matched_types = 0
    for word, idx in word_index.items():
        if idx >= vocab_size:
            continue
        vec = found.get(word)
        if vec is not None:
            matrix[idx] = vec
            matched_types += 1

    metadata = {
        "glove_path": str(glove_path),
        "dim": dim,
        "vocab_size": vocab_size,
        "matched_types": matched_types,
        "unmatched_types": vocab_size - 1 - matched_types,  # -1 excludes the padding row
        "coverage_%": round(100 * matched_types / (vocab_size - 1), 2),
        "oov_init": "independent uniform(-0.05, 0.05) per unmatched row",
        "padding_row_init": "zero (row 0)",
        "seed": seed,
        "layer_trainable": "set by the model factory (Task 4), not fixed here - "
        "data_config.glove.trainable=true applies to every row unless the "
        "Embedding layer separately masks index 0",
    }
    return matrix, metadata
