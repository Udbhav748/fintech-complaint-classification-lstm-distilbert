"""Stratified dataset split generation, persistence, and strict validation utilities.

Enforces:
- 80% train / 10% validation / 10% test stratified split
- Mutually disjoint index sets
- Complete coverage of the deduplicated dataset
- Stratified class distribution
- Zero cross-split duplicate text leakage
- Zero cross-split near-duplicate cluster leakage
- Duplicate safety (deduplication verified before split)

Splitting is group-aware. `groups` is required rather than optional because an
ungrouped split silently leaks near-duplicate templates across the boundary and
inflates every downstream metric - see src/dedup.py. Callers with nothing to
group pass `np.arange(len(df))`.

All validation functions fail loudly with SplitValidationError upon invariant violation.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data import LABEL_COL, TEXT_COL
from src.reproducibility import set_seed


class SplitValidationError(ValueError):
    """Raised when dataset split integrity invariants are violated."""
    pass


def create_stratified_split(
    df: pd.DataFrame,
    groups: np.ndarray,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    label_col: str = LABEL_COL,
    text_col: str = TEXT_COL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates a deterministic, stratified, group-aware train/val/test split.

    Rows sharing a group id always land in the same split. Groups are the
    near-duplicate clusters from src.dedup; pass np.arange(len(df)) to make every
    row its own group.

    Args:
        df: Cleaned and deduplicated DataFrame.
        groups: Group id per row. Rows in one group are never separated.
        train_ratio: Target proportion for training set (default: 0.80).
        val_ratio: Target proportion for validation set (default: 0.10).
        test_ratio: Target proportion for test set (default: 0.10).
        seed: Random seed for reproducible splitting.
        label_col: Name of product/label column for stratification.
        text_col: Name of narrative text column.

    Returns:
        (train_idx, val_idx, test_idx) as int64 NumPy arrays.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    total_ratio_parts = (train_ratio, val_ratio, test_ratio)
    n_folds = int(round(1 / min(val_ratio, test_ratio)))
    if not all(np.isclose(r * n_folds, round(r * n_folds)) for r in total_ratio_parts):
        raise ValueError(
            f"Ratios {total_ratio_parts} are not expressible as whole folds of 1/{n_folds}"
        )

    groups = np.asarray(groups)
    if len(groups) != len(df):
        raise ValueError(f"groups has {len(groups)} entries but df has {len(df)} rows")

    set_seed(seed)
    labels = df[label_col].values

    # Whole groups are dealt into n_folds stratified folds, then folds are
    # assigned to splits. Going through folds rather than two successive
    # train_test_split calls is what keeps a group from being torn in half.
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_row = np.empty(len(df), dtype=np.int64)
    for fold_id, (_, fold_rows) in enumerate(folds.split(np.zeros(len(df)), labels, groups)):
        fold_of_row[fold_rows] = fold_id

    n_val_folds = int(round(val_ratio * n_folds))
    n_test_folds = int(round(test_ratio * n_folds))
    val_folds = set(range(n_val_folds))
    test_folds = set(range(n_val_folds, n_val_folds + n_test_folds))

    val_idx = np.where(np.isin(fold_of_row, list(val_folds)))[0]
    test_idx = np.where(np.isin(fold_of_row, list(test_folds)))[0]
    train_idx = np.where(~np.isin(fold_of_row, list(val_folds | test_folds)))[0]

    train_idx = np.ascontiguousarray(np.sort(train_idx), dtype=np.int64)
    val_idx = np.ascontiguousarray(np.sort(val_idx), dtype=np.int64)
    test_idx = np.ascontiguousarray(np.sort(test_idx), dtype=np.int64)

    # Validate split immediately
    validate_split(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        groups=groups,
        expected_train_ratio=train_ratio,
        expected_val_ratio=val_ratio,
        expected_test_ratio=test_ratio,
        label_col=label_col,
        text_col=text_col,
    )

    return train_idx, val_idx, test_idx


def validate_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: np.ndarray,
    expected_train_ratio: float = 0.80,
    expected_val_ratio: float = 0.10,
    expected_test_ratio: float = 0.10,
    label_col: str = LABEL_COL,
    text_col: str = TEXT_COL,
    proportion_tolerance: float = 0.015,
) -> None:
    """Validates split integrity and fails loudly if any invariant is violated.

    Invariants Checked:
    1. Duplicate Safety: Deduplication must have occurred before split.
    2. Index Disjointness: train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅.
    3. Complete Coverage: train ∪ val ∪ test == set(range(len(df))).
    4. Cross-split Text Leakage: No identical narrative text appears across splits.
    5. Cross-split Group Leakage: No group id appears in more than one split.
    6. Proportions & Stratification: Class distribution matches dataset within tolerance.
    """
    errors: list[str] = []
    n_total = len(df)
    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)

    # 1. Duplicate Safety (pre-split deduplication check)
    dup_count = df[text_col].duplicated().sum()
    if dup_count > 0:
        errors.append(
            f"Duplicate safety violated: DataFrame contains {dup_count} duplicate narratives. "
            "Deduplication must occur before splitting."
        )

    # 2. Index Disjointness
    s_train = set(train_idx.tolist())
    s_val = set(val_idx.tolist())
    s_test = set(test_idx.tolist())

    overlap_tv = s_train.intersection(s_val)
    overlap_tt = s_train.intersection(s_test)
    overlap_vt = s_val.intersection(s_test)

    if overlap_tv:
        errors.append(f"Index disjointness violated: train ∩ val contains {len(overlap_tv)} overlapping indices.")
    if overlap_tt:
        errors.append(f"Index disjointness violated: train ∩ test contains {len(overlap_tt)} overlapping indices.")
    if overlap_vt:
        errors.append(f"Index disjointness violated: val ∩ test contains {len(overlap_vt)} overlapping indices.")

    # 3. Complete Coverage
    union_indices = s_train.union(s_val).union(s_test)
    expected_indices = set(range(n_total))
    if union_indices != expected_indices:
        missing = expected_indices - union_indices
        extra = union_indices - expected_indices
        errors.append(
            f"Complete coverage violated: train ∪ val ∪ test has {len(union_indices)} items (expected {n_total}). "
            f"Missing {len(missing)} indices, out-of-bounds {len(extra)} indices."
        )

    # 4. Cross-split Text Leakage
    train_texts = set(df.iloc[train_idx][text_col])
    val_texts = set(df.iloc[val_idx][text_col])
    test_texts = set(df.iloc[test_idx][text_col])

    leak_tv = train_texts.intersection(val_texts)
    leak_tt = train_texts.intersection(test_texts)
    leak_vt = val_texts.intersection(test_texts)

    if leak_tv:
        errors.append(f"Cross-split text leakage detected: {len(leak_tv)} duplicate narratives cross train/val.")
    if leak_tt:
        errors.append(f"Cross-split text leakage detected: {len(leak_tt)} duplicate narratives cross train/test.")
    if leak_vt:
        errors.append(f"Cross-split text leakage detected: {len(leak_vt)} duplicate narratives cross val/test.")

    # 5. Cross-split Group Leakage (near-duplicate clusters must stay whole)
    groups = np.asarray(groups)
    if len(groups) != n_total:
        errors.append(f"groups has {len(groups)} entries but df has {n_total} rows.")
    else:
        g_train = set(groups[train_idx].tolist())
        g_val = set(groups[val_idx].tolist())
        g_test = set(groups[test_idx].tolist())
        for name, shared in (
            ("train/val", g_train & g_val),
            ("train/test", g_train & g_test),
            ("val/test", g_val & g_test),
        ):
            if shared:
                errors.append(
                    f"Cross-split group leakage detected: {len(shared)} near-duplicate "
                    f"clusters cross {name}."
                )

    # 6. Split Size Proportions
    actual_train_r = n_train / n_total
    actual_val_r = n_val / n_total
    actual_test_r = n_test / n_total

    if abs(actual_train_r - expected_train_ratio) > proportion_tolerance:
        errors.append(
            f"Train split size ratio mismatch: expected ~{expected_train_ratio:.2f}, got {actual_train_r:.4f}"
        )
    if abs(actual_val_r - expected_val_ratio) > proportion_tolerance:
        errors.append(
            f"Validation split size ratio mismatch: expected ~{expected_val_ratio:.2f}, got {actual_val_r:.4f}"
        )
    if abs(actual_test_r - expected_test_ratio) > proportion_tolerance:
        errors.append(
            f"Test split size ratio mismatch: expected ~{expected_test_ratio:.2f}, got {actual_test_r:.4f}"
        )

    # 7. Stratification Class Distribution
    overall_dist = df[label_col].value_counts(normalize=True)
    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_dist = df.iloc[split_idx][label_col].value_counts(normalize=True)
        for cls_name, p_overall in overall_dist.items():
            p_split = split_dist.get(cls_name, 0.0)
            if abs(p_split - p_overall) > proportion_tolerance:
                errors.append(
                    f"Class distribution mismatch in {split_name} for '{cls_name}': "
                    f"dataset proportion={p_overall:.4f}, split proportion={p_split:.4f} (diff > {proportion_tolerance})"
                )

    if errors:
        raise SplitValidationError("Split validation failed with errors:\n" + "\n".join(f"- {e}" for e in errors))


def save_splits(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Union[str, Path] = "data/splits",
) -> None:
    """Persists split index arrays to disk as .npy files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_idx.npy", train_idx)
    np.save(out / "val_idx.npy", val_idx)
    np.save(out / "test_idx.npy", test_idx)


def load_splits(
    split_dir: Union[str, Path] = "data/splits",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads frozen split indices from disk."""
    p = Path(split_dir)
    train_path = p / "train_idx.npy"
    val_path = p / "val_idx.npy"
    test_path = p / "test_idx.npy"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Split file {path} not found. Splits must be frozen once in Stage 2.")

    train_idx = np.load(train_path)
    val_idx = np.load(val_path)
    test_idx = np.load(test_path)
    return train_idx, val_idx, test_idx


def check_cross_split_leakage(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    text_col: str = TEXT_COL,
) -> dict[str, int]:
    """Inspects and counts exact cross-split text duplicates.

    Returns:
        dict with counts of duplicate text overlap across pairs:
        {"train_val": count, "train_test": count, "val_test": count, "total_leakage": count}
    """
    train_texts = set(df.iloc[train_idx][text_col])
    val_texts = set(df.iloc[val_idx][text_col])
    test_texts = set(df.iloc[test_idx][text_col])

    leak_tv = len(train_texts.intersection(val_texts))
    leak_tt = len(train_texts.intersection(test_texts))
    leak_vt = len(val_texts.intersection(test_texts))

    return {
        "train_val": leak_tv,
        "train_test": leak_tt,
        "val_test": leak_vt,
        "total_leakage": leak_tv + leak_tt + leak_vt,
    }
