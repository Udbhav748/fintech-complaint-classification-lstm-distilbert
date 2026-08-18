"""Training-only class-weight computation for Phase 4.

Calculates inverse frequency class weights strictly on the training split to
prevent validation/test data leakage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_training_class_weights(
    train_df: pd.DataFrame,
    target_col: str = "product",
    label_mapping: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Compute balanced class weights strictly on the training partition.

    Formula:
        weight[c] = N_train / (num_classes * count[c])

    Args:
        train_df: DataFrame containing the training data only.
        target_col: Target column name.
        label_mapping: Optional integer ID to label string mapping.

    Returns:
        Dictionary containing class counts, class weights by name and ID, and metadata.
    """
    total_train = len(train_df)
    class_counts = train_df[target_col].value_counts().to_dict()
    classes = sorted(class_counts.keys())
    num_classes = len(classes)

    weights_by_name: dict[str, float] = {}
    for cls_name in classes:
        cnt = class_counts[cls_name]
        w = total_train / (num_classes * cnt) if cnt > 0 else 0.0
        weights_by_name[cls_name] = round(float(w), 6)

    # Reversible integer mapping
    if label_mapping is not None:
        name_to_id = {v: k for k, v in label_mapping.items()}
        weights_by_id = {str(name_to_id[cls_name]): weights_by_name[cls_name] for cls_name in classes if cls_name in name_to_id}
    else:
        weights_by_id = {str(i): weights_by_name[cls_name] for i, cls_name in enumerate(classes)}

    # Measure max to min ratio
    min_cnt = min(class_counts.values())
    max_cnt = max(class_counts.values())
    imbalance_ratio = round(max_cnt / min_cnt, 3) if min_cnt > 0 else 0.0

    logger.info("Training class weights computed on %d rows: %s", total_train, weights_by_name)
    logger.info("Training class imbalance ratio: %.3fx (max: %d, min: %d)", imbalance_ratio, max_cnt, min_cnt)

    return {
        "total_training_samples": total_train,
        "num_classes": num_classes,
        "imbalance_ratio": imbalance_ratio,
        "class_counts": class_counts,
        "weights_by_name": weights_by_name,
        "weights_by_id": weights_by_id,
        "formula": "N_train / (num_classes * class_count)",
        "note": "Computed strictly on the training partition. Because acquisition intentionally collected ~20k rows per product, class weights are close to 1.0 (1.19x ratio) and are expected to produce minimal delta in E3.",
    }
