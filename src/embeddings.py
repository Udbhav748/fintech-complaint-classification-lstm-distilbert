"""GloVe vocabulary lookup and coverage measurement."""

from pathlib import Path

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
