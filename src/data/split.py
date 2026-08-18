"""Group-aware stratified data splitting for Phase 4.

Prevents exact duplicate narratives and conflicting-label groups from crossing
train, validation, and test boundaries while preserving approximately balanced
class distributions across all splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitSummary:
    """Summary of data split results and validation checks."""

    total_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_groups: int
    val_groups: int
    test_groups: int
    class_distribution: dict[str, dict[str, Any]]
    leakage_check_passed: bool
    complaint_id_overlap: int
    group_id_overlap: int


def group_stratified_split(
    df: pd.DataFrame,
    group_col: str = "narrative_group_id",
    target_col: str = "product",
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a reproducible, group-aware stratified train/val/test split.

    Guarantees:
      1. Every narrative group ID is placed entirely within ONE partition.
      2. Conflicting duplicate groups are wholly contained within ONE partition.
      3. No Complaint ID or narrative text crosses split boundaries.
      4. Target split ratios (e.g. 70/15/15) are maintained across the 5 classes.
    """
    train_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got: {ratios}")

    # Unique classes and mapping
    classes = sorted(df[target_col].unique())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    num_classes = len(classes)

    # Compute group-level class histograms
    group_records: list[dict[str, Any]] = []
    rng = np.random.RandomState(random_seed)

    grouped = df.groupby(group_col)
    for gid, group_df in grouped:
        counts = np.zeros(num_classes, dtype=int)
        for cls_name, cnt in group_df[target_col].value_counts().items():
            counts[class_to_idx[cls_name]] = cnt
        total_in_group = len(group_df)
        group_records.append(
            {
                "group_id": gid,
                "size": total_in_group,
                "counts": counts,
                "random_tiebreaker": rng.rand(),
            }
        )

    # Total counts per class across the entire dataset
    total_class_counts = np.zeros(num_classes, dtype=float)
    for rec in group_records:
        total_class_counts += rec["counts"]

    # Target counts per partition
    target_counts = {
        "train": total_class_counts * train_ratio,
        "val": total_class_counts * val_ratio,
        "test": total_class_counts * test_ratio,
    }

    # Sort groups: largest groups first for stable bin-packing, then randomized tie-breaker
    group_records.sort(key=lambda r: (-r["size"], r["random_tiebreaker"]))

    target_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    allocated_counts = {
        "train": np.zeros(num_classes, dtype=float),
        "val": np.zeros(num_classes, dtype=float),
        "test": np.zeros(num_classes, dtype=float),
    }
    group_assignments: dict[str, str] = {}

    for rec in group_records:
        gid = rec["group_id"]
        g_cnt = rec["counts"]

        best_partition = None
        best_cost = float("inf")

        for part, p_ratio in target_ratios.items():
            # Normalized share of each class if allocated to partition part
            current_share = (allocated_counts[part] + g_cnt) / (p_ratio * total_class_counts)
            cost = float(np.sum(current_share[g_cnt > 0]))

            if cost < best_cost:
                best_cost = cost
                best_partition = part

        # Assign group to the best partition
        group_assignments[gid] = best_partition
        allocated_counts[best_partition] += g_cnt


    # Map partition assignments back to DataFrame
    df_out = df.copy()
    df_out["split"] = df_out[group_col].map(group_assignments)

    train_df = df_out[df_out["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val_df = df_out[df_out["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    test_df = df_out[df_out["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

    # Perform strict leakage validation
    validate_split_invariants(train_df, val_df, test_df, group_col=group_col, id_col="complaint_id")

    return train_df, val_df, test_df


def validate_split_invariants(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "narrative_group_id",
    id_col: str = "complaint_id",
) -> None:
    """Assert that no Complaint ID or narrative group leaks across train/val/test."""
    train_ids = set(train_df[id_col])
    val_ids = set(val_df[id_col])
    test_ids = set(test_df[id_col])

    id_overlaps = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if id_overlaps:
        raise ValueError(f"Data leakage detected! Overlapping {id_col} across splits: {len(id_overlaps)} IDs")

    train_groups = set(train_df[group_col])
    val_groups = set(val_df[group_col])
    test_groups = set(test_df[group_col])

    group_overlaps = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    if group_overlaps:
        raise ValueError(
            f"Data leakage detected! Overlapping narrative groups across splits: {len(group_overlaps)} groups"
        )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            raise ValueError(f"Split {name} is empty!")

    logger.info("Split validation PASSED: 0 ID overlap, 0 narrative group overlap across all partitions.")
