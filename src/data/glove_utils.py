"""GloVe embedding download and vocabulary coverage analysis for Phase 4.

Downloads and extracts GloVe embeddings (GloVe 6B 100d) and computes coverage
metrics strictly against the training-split LSTM vocabulary.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.vocabulary import Vocabulary, tokenize_text

logger = logging.getLogger(__name__)

GLOVE_ZIP_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_FALLBACK_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"


def download_and_extract_glove(
    target_dim: int = 100,
    cache_dir: Path | str = "data/embeddings",
) -> Path:
    """Download GloVe 6B zip archive if not present and extract target dimension text file.

    Args:
        target_dim: Embedding dimension (e.g. 100).
        cache_dir: Directory where embeddings are stored.

    Returns:
        Path to extracted glove.6B.{target_dim}d.txt file.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    target_file = cache_path / f"glove.6B.{target_dim}d.txt"
    if target_file.exists() and target_file.stat().st_size > 0:
        logger.info("GloVe %dd file already exists: %s", target_dim, target_file)
        return target_file

    zip_file = cache_path / "glove.6B.zip"
    target_member = f"glove.6B.{target_dim}d.txt"

    # Download zip if not already downloaded
    if not zip_file.exists() or zip_file.stat().st_size == 0:
        logger.info("Downloading GloVe archive from %s...", GLOVE_ZIP_URL)
        try:
            urllib.request.urlretrieve(GLOVE_ZIP_URL, zip_file)
        except Exception as e:
            logger.warning("Primary download failed (%s), trying Stanford fallback...", e)
            urllib.request.urlretrieve(GLOVE_FALLBACK_URL, zip_file)

    logger.info("Extracting %s from %s...", target_member, zip_file)
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extract(target_member, path=cache_path)

    # Remove the 862MB zip file to keep disk space minimal
    if zip_file.exists():
        try:
            zip_file.unlink()
            logger.info("Cleaned up temporary zip archive: %s", zip_file)
        except Exception as e:
            logger.warning("Could not delete zip file: %s", e)

    if not target_file.exists():
        raise FileNotFoundError(f"Failed to extract {target_member} to {cache_path}")

    logger.info("GloVe %dd successfully prepared at %s (size: %.2f MB)", target_dim, target_file, target_file.stat().st_size / (1024 * 1024))
    return target_file



def load_glove_vocabulary(glove_path: Path | str) -> set[str]:
    """Read word vocabulary from GloVe text file without loading full float vectors into memory."""
    glove_file = Path(glove_path)
    if not glove_file.exists():
        raise FileNotFoundError(f"GloVe file not found: {glove_file}")

    glove_words: set[str] = set()
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if parts:
                glove_words.add(parts[0])

    logger.info("Loaded %d words from GloVe vocabulary: %s", len(glove_words), glove_file.name)
    return glove_words


def analyze_glove_coverage(
    vocab: Vocabulary,
    glove_path: Path | str,
    train_texts: list[str],
    target_dim: int = 100,
) -> dict[str, Any]:
    """Analyze GloVe coverage strictly on the training vocabulary.

    Args:
        vocab: Fitted training Vocabulary.
        glove_path: Path to GloVe .txt file.
        train_texts: Training narrative texts.
        target_dim: Embedding dimension.

    Returns:
        Dictionary containing coverage statistics, top missing words, and sample matches.
    """
    glove_words = load_glove_vocabulary(glove_path)

    # Exclude special tokens <pad> and <unk> from word coverage calculation
    special_tokens = {vocab.pad_token, vocab.unk_token}
    model_words = [w for w in vocab.word2idx.keys() if w not in special_tokens]

    matched_words: list[str] = []
    missing_words: list[tuple[str, int]] = []

    for w in model_words:
        cnt = vocab.word_counts.get(w, 0)
        if w in glove_words:
            matched_words.append(w)
        else:
            missing_words.append((w, cnt))

    # Sort missing words by training frequency descending
    missing_words.sort(key=lambda x: -x[1])

    # Compute token-level frequency coverage across all training tokens
    total_train_tokens = sum(vocab.word_counts.values())
    covered_train_tokens = sum(vocab.word_counts.get(w, 0) for w in matched_words)

    vocab_coverage_pct = (len(matched_words) / len(model_words) * 100.0) if model_words else 0.0
    token_coverage_pct = (covered_train_tokens / total_train_tokens * 100.0) if total_train_tokens > 0 else 0.0
    oov_rate_pct = 100.0 - vocab_coverage_pct

    logger.info(
        "GloVe Coverage on Training Vocab: %d/%d words (%.2f%%), Token Frequency Coverage: %.2f%%, OOV Rate: %.2f%%",
        len(matched_words),
        len(model_words),
        vocab_coverage_pct,
        token_coverage_pct,
        oov_rate_pct,
    )

    top_missing_summary = [
        {"word": w, "train_frequency": c} for w, c in missing_words[:30]
    ]

    return {
        "embedding_name": f"glove.6B.{target_dim}d",
        "dimension": target_dim,
        "glove_total_words": len(glove_words),
        "model_vocab_words": len(model_words),
        "matched_words_count": len(matched_words),
        "missing_words_count": len(missing_words),
        "vocab_coverage_percentage": round(vocab_coverage_pct, 2),
        "token_frequency_coverage_percentage": round(token_coverage_pct, 2),
        "oov_rate_percentage": round(oov_rate_pct, 2),
        "top_missing_words": top_missing_summary,
        "sample_matched_words": matched_words[:20],
        "note": "Evaluated strictly against the training vocabulary. Missing words include CFPB redaction token 'xxxx', specialized financial institutions/servicers (e.g. MOHELA, ChexSystems, Zelle), and domain-specific terms.",
    }
