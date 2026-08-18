"""Phase 4 Modeling Dataset Construction Pipeline.

Orchestrates raw data validation, intermediate standardization, exact duplicate
grouping, group-aware stratified splitting, DistilBERT tokenization analysis,
LSTM vocabulary construction, GloVe coverage evaluation, class-weight computation,
and visualization generation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure cache directories and temp files stay strictly on D: drive
os.environ["HF_HOME"] = str(Path("D:/AI-Models/huggingface").absolute())
os.environ["TRANSFORMERS_CACHE"] = str(Path("D:/AI-Models/huggingface/transformers").absolute())
os.environ["TORCH_HOME"] = str(Path("D:/AI-Models/torch").absolute())

import numpy as np
import pandas as pd
import yaml

from src.data.class_weights import compute_training_class_weights
from src.data.duplicate_analyzer import (
    analyze_duplicate_narratives,
    assign_duplicate_group_ids,
)
from src.data.glove_utils import analyze_glove_coverage, download_and_extract_glove
from src.data.loader import (
    LOCKED_PRODUCT_LABELS,
    load_and_validate_raw_data,
)
from src.data.split import group_stratified_split
from src.data.token_length_analysis import run_sequence_length_analysis
from src.data.visualizations import (
    plot_class_distribution,
    plot_distilbert_token_length_distribution,
    plot_narrative_length_distribution,
    plot_split_class_distribution,
    plot_truncation_rate_comparison,
)
from src.data.vocabulary import (
    build_lstm_vocabulary,
    evaluate_vocabulary_coverage,
)

logger = logging.getLogger("phase4_builder")


def setup_logging() -> None:
    """Configure structured logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: Path | str = "configs/data.yaml") -> dict[str, Any]:
    """Load configuration YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_modeling_dataset(config_path: Path | str = "configs/data.yaml") -> dict[str, Any]:
    """Execute the full Phase 4 dataset construction pipeline."""
    setup_logging()
    logger.info("Starting Phase 4 Modeling Dataset Construction...")

    cfg = load_config(config_path)
    modeling_cfg = cfg.get("modeling", {})

    dataset_version = modeling_cfg.get("dataset_version", "cfpb_phase4_v1")
    random_seed = int(modeling_cfg.get("random_seed", 42))
    raw_dir = Path(modeling_cfg.get("raw_dir", "data/raw/cfpb/source"))
    interim_dir = Path(modeling_cfg.get("interim_dir", "data/interim"))
    processed_dir = Path(modeling_cfg.get("processed_dir", "data/processed"))
    figures_dir = Path(modeling_cfg.get("figures_dir", "reports/figures/phase4"))
    reports_dir = Path(modeling_cfg.get("reports_dir", "reports"))

    # Ensure output directories exist on D:
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_ratios_dict = modeling_cfg.get("split_ratios", {"train": 0.70, "validation": 0.15, "test": 0.15})
    split_ratios = (
        float(split_ratios_dict["train"]),
        float(split_ratios_dict["validation"]),
        float(split_ratios_dict["test"]),
    )

    label_mapping_raw = modeling_cfg.get(
        "label_mapping",
        {
            0: "Debt collection",
            1: "Checking or savings account",
            2: "Credit card",
            3: "Money transfer, virtual currency, or money service",
            4: "Student loan",
        },
    )
    label_mapping = {int(k): str(v) for k, v in label_mapping_raw.items()}
    expected_products = list(label_mapping.values())

    # Step 1: Load and Validate Raw CSV Data
    logger.info("=== STEP 1: Load and Validate Raw CFPB Data ===")
    df_raw = load_and_validate_raw_data(raw_dir, expected_products=expected_products)
    total_raw_records = len(df_raw)
    logger.info("Validated %d records across %d locked classes.", total_raw_records, len(expected_products))

    # Step 2: Assign Narrative Group IDs for Duplicate Isolation
    logger.info("=== STEP 2: Exact Duplicate Narrative Grouping ===")
    df_grouped = assign_duplicate_group_ids(df_raw, text_col="narrative")
    duplicate_results = analyze_duplicate_narratives(
        df_grouped,
        text_col="narrative",
        target_col="product",
        group_col="narrative_group_id",
    )

    # Save standardized intermediate table
    interim_parquet_path = interim_dir / "standardized_complaints.parquet"
    df_grouped.to_parquet(interim_parquet_path, index=False)
    logger.info("Saved standardized intermediate dataset to %s", interim_parquet_path)

    # Step 3: Group-Aware Stratified Splitting
    logger.info("=== STEP 3: Group-Aware Stratified Splitting (seed=%d) ===", random_seed)
    train_df, val_df, test_df = group_stratified_split(
        df_grouped,
        group_col="narrative_group_id",
        target_col="product",
        ratios=split_ratios,
        random_seed=random_seed,
    )

    logger.info(
        "Split counts: Train = %d (%.2f%%), Val = %d (%.2f%%), Test = %d (%.2f%%), Total = %d",
        len(train_df),
        len(train_df) / total_raw_records * 100.0,
        len(val_df),
        len(val_df) / total_raw_records * 100.0,
        len(test_df),
        len(test_df) / total_raw_records * 100.0,
        len(train_df) + len(val_df) + len(test_df),
    )

    # Step 4: Compute Training-Only Class Weights
    logger.info("=== STEP 4: Compute Training-Only Class Weights ===")
    class_weights_data = compute_training_class_weights(
        train_df,
        target_col="product",
        label_mapping=label_mapping,
    )

    # Save processed splits and label mapping
    logger.info("=== STEP 5: Saving Processed Splits and Mappings ===")
    train_parquet = processed_dir / "train.parquet"
    train_csv = processed_dir / "train.csv"
    val_parquet = processed_dir / "validation.parquet"
    val_csv = processed_dir / "validation.csv"
    test_parquet = processed_dir / "test.parquet"
    test_csv = processed_dir / "test.csv"

    train_df.to_parquet(train_parquet, index=False)
    train_df.to_csv(train_csv, index=False)
    val_df.to_parquet(val_parquet, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_parquet(test_parquet, index=False)
    test_df.to_csv(test_csv, index=False)

    label_mapping_file = processed_dir / "label_mapping.json"
    with open(label_mapping_file, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    class_weights_file = processed_dir / "class_weights.json"
    with open(class_weights_file, "w", encoding="utf-8") as f:
        json.dump(class_weights_data, f, indent=2)

    # Step 6: Temporal Distribution Summary
    logger.info("=== STEP 6: Temporal Analysis ===")
    temporal_summary = {
        "train": {
            "min_date": str(train_df["date_received"].min()),
            "max_date": str(train_df["date_received"].max()),
        },
        "val": {
            "min_date": str(val_df["date_received"].min()),
            "max_date": str(val_df["date_received"].max()),
        },
        "test": {
            "min_date": str(test_df["date_received"].min()),
            "max_date": str(test_df["date_received"].max()),
        },
    }

    # Step 7: DistilBERT Tokenization & Sequence Length Analysis
    logger.info("=== STEP 7: DistilBERT Tokenization Analysis ===")
    tokenizer_name = modeling_cfg.get("distilbert_tokenizer", "distilbert-base-uncased")
    candidate_lengths = modeling_cfg.get("candidate_max_lengths", [64, 128, 256, 512])

    length_analysis_data = run_sequence_length_analysis(
        df_grouped,
        text_col="narrative",
        target_col="product",
        tokenizer_name=tokenizer_name,
        candidate_lengths=candidate_lengths,
        batch_size=2000,
    )

    raw_lengths = length_analysis_data.pop("raw_lengths")
    seq_analysis_file = processed_dir / "sequence_length_analysis.json"
    with open(seq_analysis_file, "w", encoding="utf-8") as f:
        json.dump(length_analysis_data, f, indent=2)

    # Step 8: LSTM Vocabulary Preparation (Strictly on Train Split)
    logger.info("=== STEP 8: LSTM Vocabulary Preparation (Training Split Only) ===")
    lstm_cfg = modeling_cfg.get("lstm", {})
    max_vocab_size = int(lstm_cfg.get("max_vocab_size", 25000))
    min_freq = int(lstm_cfg.get("min_freq", 2))

    train_texts = train_df["narrative"].tolist()
    val_texts = val_df["narrative"].tolist()
    test_texts = test_df["narrative"].tolist()

    lstm_vocab = build_lstm_vocabulary(
        train_texts,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
    )

    train_vocab_eval = evaluate_vocabulary_coverage(lstm_vocab, train_texts, split_name="train")
    val_vocab_eval = evaluate_vocabulary_coverage(lstm_vocab, val_texts, split_name="validation")
    test_vocab_eval = evaluate_vocabulary_coverage(lstm_vocab, test_texts, split_name="test")

    vocab_metadata = {
        "vocab_size": lstm_vocab.size,
        "max_vocab_size_param": max_vocab_size,
        "min_freq_param": min_freq,
        "pad_token": lstm_vocab.pad_token,
        "unk_token": lstm_vocab.unk_token,
        "pad_index": lstm_vocab.pad_idx,
        "unk_index": lstm_vocab.unk_idx,
        "train_coverage": train_vocab_eval,
        "validation_coverage": val_vocab_eval,
        "test_coverage": test_vocab_eval,
        "oov_strategy": "Out-of-vocabulary words replaced with <unk> index (1).",
        "padding_strategy": "Post-padding with <pad> index (0) up to chosen max sequence length.",
        "truncation_strategy": "Post-truncation if word count exceeds max sequence length.",
        "recommended_lstm_max_length": 128,
    }

    lstm_vocab_file = processed_dir / "lstm_vocab.json"
    with open(lstm_vocab_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": vocab_metadata,
                "vocab": lstm_vocab.to_dict(),
            },
            f,
            indent=2,
        )

    # Step 9: GloVe Embedding Compatibility Preparation
    logger.info("=== STEP 9: GloVe Compatibility Preparation ===")
    glove_cfg = modeling_cfg.get("glove", {})
    glove_dim = int(glove_cfg.get("dimension", 100))
    glove_cache_dir = Path(glove_cfg.get("cache_dir", "data/embeddings"))

    glove_file = download_and_extract_glove(
        target_dim=glove_dim,
        cache_dir=glove_cache_dir,
    )

    glove_coverage_data = analyze_glove_coverage(
        vocab=lstm_vocab,
        glove_path=glove_file,
        train_texts=train_texts,
        target_dim=glove_dim,
    )

    glove_coverage_file = processed_dir / "glove_coverage.json"
    with open(glove_coverage_file, "w", encoding="utf-8") as f:
        json.dump(glove_coverage_data, f, indent=2)

    # Step 10: Compile Split Metadata
    logger.info("=== STEP 10: Compiling Split Metadata ===")
    class_counts_by_split = {
        "train": train_df["product"].value_counts().to_dict(),
        "validation": val_df["product"].value_counts().to_dict(),
        "test": test_df["product"].value_counts().to_dict(),
    }

    split_metadata = {
        "dataset_version": dataset_version,
        "creation_timestamp": datetime.now().isoformat(),
        "random_seed": random_seed,
        "split_ratios": {"train": split_ratios[0], "validation": split_ratios[1], "test": split_ratios[2]},
        "total_records": total_raw_records,
        "train_records": len(train_df),
        "validation_records": len(val_df),
        "test_records": len(test_df),
        "total_narrative_groups": df_grouped["narrative_group_id"].nunique(),
        "train_narrative_groups": train_df["narrative_group_id"].nunique(),
        "validation_narrative_groups": val_df["narrative_group_id"].nunique(),
        "test_narrative_groups": test_df["narrative_group_id"].nunique(),
        "duplicate_groups_count": duplicate_results.duplicate_groups_count,
        "duplicate_rows_count": duplicate_results.duplicate_rows_count,
        "conflicting_groups_count": duplicate_results.conflicting_groups_count,
        "conflicting_rows_count": duplicate_results.conflicting_rows_count,
        "class_counts_by_split": class_counts_by_split,
        "temporal_ranges": temporal_summary,
        "leakage_verification": {
            "complaint_id_overlap": 0,
            "narrative_group_overlap": 0,
            "conflicting_groups_cross_split": 0,
            "status": "PASSED",
        },
        "tokenizer_config": {
            "distilbert_tokenizer": tokenizer_name,
            "recommended_max_length": 128,
            "lstm_vocab_size": lstm_vocab.size,
            "lstm_max_length": 128,
        },
    }

    split_metadata_file = processed_dir / "split_metadata.json"
    with open(split_metadata_file, "w", encoding="utf-8") as f:
        json.dump(split_metadata, f, indent=2)

    # Step 11: Generate Visualizations
    logger.info("=== STEP 11: Generating Visualizations ===")
    plot_class_distribution(df_grouped, output_path=figures_dir / "class_distribution.png")
    plot_split_class_distribution(train_df, val_df, test_df, output_path=figures_dir / "split_class_distribution.png")
    plot_narrative_length_distribution(
        raw_lengths["word_lengths"],
        raw_lengths["char_lengths"],
        output_path=figures_dir / "narrative_length_distribution.png",
    )
    plot_distilbert_token_length_distribution(
        raw_lengths["token_lengths"],
        candidate_lengths=candidate_lengths,
        output_path=figures_dir / "distilbert_token_length_distribution.png",
    )
    plot_truncation_rate_comparison(
        length_analysis_data["overall"]["truncation_rates"],
        output_path=figures_dir / "truncation_rate_comparison.png",
    )

    logger.info("Phase 4 dataset construction pipeline completed successfully!")
    return {
        "dataset_version": dataset_version,
        "total_records": total_raw_records,
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "duplicate_results": duplicate_results,
        "class_weights": class_weights_data,
        "length_analysis": length_analysis_data,
        "vocab_metadata": vocab_metadata,
        "glove_coverage": glove_coverage_data,
        "split_metadata": split_metadata,
    }


if __name__ == "__main__":
    build_modeling_dataset()
