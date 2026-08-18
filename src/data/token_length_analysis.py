"""Sequence length and DistilBERT tokenization analysis for Phase 4.

Measures character length, word count, and actual DistilBERT token count across
the modeling dataset. Evaluates truncation rates for candidate max_length settings.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def compute_length_statistics(lengths: list[int] | np.ndarray) -> dict[str, float]:
    """Compute summary statistics and percentiles for a 1D sequence of lengths."""
    arr = np.array(lengths, dtype=float)
    if len(arr) == 0:
        return {}
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def compute_truncation_rates(
    lengths: list[int] | np.ndarray,
    candidate_lengths: list[int] = [64, 128, 256, 512],
) -> dict[str, dict[str, Any]]:
    """Compute exact truncation counts and percentages for candidate max_lengths."""
    arr = np.array(lengths)
    total = len(arr)
    results: dict[str, dict[str, Any]] = {}
    for max_len in candidate_lengths:
        truncated_count = int(np.sum(arr > max_len))
        truncated_pct = float(truncated_count / total * 100.0) if total > 0 else 0.0
        results[str(max_len)] = {
            "max_length": max_len,
            "truncated_count": truncated_count,
            "truncated_percentage": round(truncated_pct, 2),
            "retained_percentage": round(100.0 - truncated_pct, 2),
        }
    return results


def run_sequence_length_analysis(
    df: pd.DataFrame,
    text_col: str = "narrative",
    target_col: str = "product",
    tokenizer_name: str = "distilbert-base-uncased",
    candidate_lengths: list[int] = [64, 128, 256, 512],
    batch_size: int = 2000,
) -> dict[str, Any]:
    """Analyze character lengths, word counts, and DistilBERT token lengths."""
    logger.info("Computing character and word counts for %d narratives...", len(df))

    texts = df[text_col].tolist()
    char_lengths = [len(t) for t in texts]
    word_lengths = [len(t.split()) for t in texts]

    logger.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info("Tokenizing %d narratives in batches of %d...", len(texts), batch_size)
    token_lengths: list[int] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        # Tokenize without truncation or padding to measure real sequence lengths
        encoded = tokenizer(
            batch_texts,
            truncation=False,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        batch_lens = [len(ids) for ids in encoded["input_ids"]]
        token_lengths.extend(batch_lens)

    # Attach lengths to df for per-class breakdown
    analysis_df = df.copy()
    analysis_df["char_len"] = char_lengths
    analysis_df["word_len"] = word_lengths
    analysis_df["token_len"] = token_lengths

    # Global statistics
    overall_char_stats = compute_length_statistics(char_lengths)
    overall_word_stats = compute_length_statistics(word_lengths)
    overall_token_stats = compute_length_statistics(token_lengths)
    overall_truncation = compute_truncation_rates(token_lengths, candidate_lengths)

    # Per-class token statistics
    per_class_stats: dict[str, Any] = {}
    for product_name, group in analysis_df.groupby(target_col):
        cls_tokens = group["token_len"].values
        per_class_stats[str(product_name)] = {
            "token_stats": compute_length_statistics(cls_tokens),
            "truncation_rates": compute_truncation_rates(cls_tokens, candidate_lengths),
        }

    logger.info(
        "Token length analysis complete. Median tokens: %.1f, 75th: %.1f, 90th: %.1f, 95th: %.1f, 99th: %.1f",
        overall_token_stats["median"],
        overall_token_stats["p75"],
        overall_token_stats["p90"],
        overall_token_stats["p95"],
        overall_token_stats["p99"],
    )

    return {
        "tokenizer_name": tokenizer_name,
        "total_records": len(df),
        "overall": {
            "character_length": overall_char_stats,
            "word_count": overall_word_stats,
            "distilbert_token_count": overall_token_stats,
            "truncation_rates": overall_truncation,
        },
        "per_class": per_class_stats,
        "raw_lengths": {
            "char_lengths": char_lengths,
            "word_lengths": word_lengths,
            "token_lengths": token_lengths,
        },
    }
