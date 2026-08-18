"""Exact duplicate narrative and conflicting-label analysis for Phase 4.

Assigns stable group IDs to identical narrative texts and audits conflicting
product labels across duplicate groups.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConflictingDuplicateGroup:
    """Represents a duplicate narrative group that has more than one distinct Product label."""

    group_id: str
    narrative_snippet: str
    total_records: int
    label_counts: dict[str, int]


@dataclass
class DuplicateAnalysisResult:
    """Summary of exact duplicate narrative structure across the dataset."""

    total_records: int
    unique_narratives: int
    duplicate_rows_count: int
    duplicate_groups_count: int
    max_duplicate_group_size: int
    conflicting_groups_count: int
    conflicting_rows_count: int
    conflicting_groups: list[ConflictingDuplicateGroup]


def compute_narrative_group_id(text: str) -> str:
    """Compute a deterministic, stable group ID from normalized narrative text.

    Uses SHA-256 hash of the exact stripped narrative string.
    """
    normalized = text.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def assign_duplicate_group_ids(df: pd.DataFrame, text_col: str = "narrative") -> pd.DataFrame:
    """Assign a deterministic `narrative_group_id` column to every row in the DataFrame."""
    df_out = df.copy()
    # Vectorized / cached hash calculation
    unique_texts = df_out[text_col].unique()
    text_to_group_id = {text: compute_narrative_group_id(text) for text in unique_texts}
    df_out["narrative_group_id"] = df_out[text_col].map(text_to_group_id)
    return df_out


def analyze_duplicate_narratives(
    df: pd.DataFrame,
    text_col: str = "narrative",
    target_col: str = "product",
    group_col: str = "narrative_group_id",
) -> DuplicateAnalysisResult:
    """Analyze exact duplicate narrative groups and conflicting-label groups."""
    total_records = len(df)
    unique_narratives = df[text_col].nunique()

    # Group by narrative_group_id
    group_stats = df.groupby(group_col).agg(
        row_count=(text_col, "count"),
        distinct_products=(target_col, "nunique"),
        product_list=(target_col, list),
        sample_text=(text_col, "first"),
    )

    duplicate_groups = group_stats[group_stats["row_count"] > 1]
    duplicate_groups_count = len(duplicate_groups)
    duplicate_rows_count = int(duplicate_groups["row_count"].sum())
    max_duplicate_group_size = int(group_stats["row_count"].max()) if not group_stats.empty else 0

    conflicting_groups_df = group_stats[group_stats["distinct_products"] > 1]
    conflicting_groups_count = len(conflicting_groups_df)
    conflicting_rows_count = int(conflicting_groups_df["row_count"].sum())

    conflicting_groups_list: list[ConflictingDuplicateGroup] = []
    for gid, row in conflicting_groups_df.iterrows():
        # Count frequency of each product in this conflicting group
        product_counts = pd.Series(row["product_list"]).value_counts().to_dict()
        sample_snippet = row["sample_text"][:120].replace("\n", " ") + ("..." if len(row["sample_text"]) > 120 else "")
        conflicting_groups_list.append(
            ConflictingDuplicateGroup(
                group_id=str(gid),
                narrative_snippet=sample_snippet,
                total_records=int(row["row_count"]),
                label_counts=product_counts,
            )
        )

    logger.info(
        "Duplicate analysis: %d total rows, %d unique texts, %d duplicate rows across %d groups (max group: %d rows), %d conflicting groups (%d rows)",
        total_records,
        unique_narratives,
        duplicate_rows_count,
        duplicate_groups_count,
        max_duplicate_group_size,
        conflicting_groups_count,
        conflicting_rows_count,
    )

    return DuplicateAnalysisResult(
        total_records=total_records,
        unique_narratives=unique_narratives,
        duplicate_rows_count=duplicate_rows_count,
        duplicate_groups_count=duplicate_groups_count,
        max_duplicate_group_size=max_duplicate_group_size,
        conflicting_groups_count=conflicting_groups_count,
        conflicting_rows_count=conflicting_rows_count,
        conflicting_groups=conflicting_groups_list,
    )
