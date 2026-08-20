"""Dataset and split fingerprinting utilities.

Provides explicit, deterministic hashing and verification for raw, deduplicated,
and split datasets to ensure complete provenance and reproducibility across
experiments.
"""

from dataclasses import asdict, dataclass
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.data import ID_COL, LABEL_COL, TEXT_COL


class DatasetFingerprintMismatchError(ValueError):
    """Raised when a dataset does not match its expected fingerprint."""
    pass


class SplitFingerprintMismatchError(ValueError):
    """Raised when a dataset split does not match its expected fingerprint or violates split invariants."""
    pass


@dataclass
class DatasetFingerprint:
    """Explicit cryptographic and content summary of a dataset."""
    file_path: Optional[str]
    file_size_bytes: Optional[int]
    file_sha256: Optional[str]
    content_sha256: str
    total_rows: int
    num_columns: int
    columns: list[str]
    dtypes: dict[str, str]
    null_counts: dict[str, int]
    unique_narratives: int
    duplicate_narratives_count: int
    class_distribution: dict[str, int]
    class_ratios: dict[str, float]
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetFingerprint":
        return cls(**data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DatasetFingerprint":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset fingerprint manifest not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class SplitFingerprint:
    """Explicit cryptographic and integrity summary of a dataset split."""
    dataset_content_sha256: str
    train_indices_sha256: str
    val_indices_sha256: str
    test_indices_sha256: str
    train_count: int
    val_count: int
    test_count: int
    total_count: int
    train_class_distribution: dict[str, int]
    val_class_distribution: dict[str, int]
    test_class_distribution: dict[str, int]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    disjoint_indices_verified: bool
    zero_duplicate_cross_split_verified: bool
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitFingerprint":
        return cls(**data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SplitFingerprint":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Split fingerprint manifest not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_file_sha256(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file on disk in streaming chunks."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_array_sha256(arr: Union[np.ndarray, list[int]]) -> str:
    """Computes SHA-256 hash of an index array (contiguous int64 byte representation)."""
    np_arr = np.ascontiguousarray(arr, dtype=np.int64)
    return hashlib.sha256(np_arr.tobytes()).hexdigest()


def compute_dataframe_content_hash(
    df: pd.DataFrame,
    id_col: Optional[str] = ID_COL,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
) -> str:
    """Computes a deterministic content SHA-256 hash of key columns.

    This hash is invariant to Parquet file metadata or compression engine differences.
    If `id_col` is present in `df`, records are hashed in order of `id_col` for consistency.
    """
    hasher = hashlib.sha256()
    
    if id_col and id_col in df.columns:
        sorted_df = df.sort_values(by=id_col).reset_index(drop=True)
        id_series = sorted_df[id_col].astype(str)
        hasher.update(f"ID_COL:{id_col}\n".encode("utf-8"))
        for val in id_series:
            hasher.update(val.encode("utf-8") + b"\n")
    else:
        sorted_df = df

    # Hash Text Column
    hasher.update(f"TEXT_COL:{text_col}\n".encode("utf-8"))
    for val in sorted_df[text_col].astype(str):
        hasher.update(val.encode("utf-8") + b"\n")

    # Hash Label Column
    hasher.update(f"LABEL_COL:{label_col}\n".encode("utf-8"))
    for val in sorted_df[label_col].astype(str):
        hasher.update(val.encode("utf-8") + b"\n")

    return hasher.hexdigest()


def create_dataset_fingerprint(
    df_or_path: Union[pd.DataFrame, str, Path],
    id_col: Optional[str] = ID_COL,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
) -> DatasetFingerprint:
    """Generates a complete DatasetFingerprint for a DataFrame or file path."""
    file_path_str: Optional[str] = None
    file_size_bytes: Optional[int] = None
    file_sha256: Optional[str] = None

    if isinstance(df_or_path, (str, Path)):
        p = Path(df_or_path)
        file_path_str = str(p)
        file_size_bytes = p.stat().st_size
        file_sha256 = compute_file_sha256(p)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix == ".csv":
            df = pd.read_csv(p, dtype=str)
        else:
            raise ValueError(f"Unsupported file format: {p.suffix}")
    else:
        df = df_or_path

    # Compute content hash
    content_sha256 = compute_dataframe_content_hash(
        df, id_col=id_col if (id_col and id_col in df.columns) else None, text_col=text_col, label_col=label_col
    )

    # Class distribution & ratios
    class_counts = {str(k): int(v) for k, v in df[label_col].value_counts().items()}
    total_len = len(df)
    class_ratios = {k: round(v / total_len, 4) for k, v in class_counts.items()}

    # Nulls & duplicates
    null_counts = {str(col): int(df[col].isna().sum()) for col in df.columns}
    unique_narratives = int(df[text_col].nunique())
    duplicate_narratives_count = total_len - unique_narratives

    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    return DatasetFingerprint(
        file_path=file_path_str,
        file_size_bytes=file_size_bytes,
        file_sha256=file_sha256,
        content_sha256=content_sha256,
        total_rows=total_len,
        num_columns=len(df.columns),
        columns=[str(c) for c in df.columns],
        dtypes=dtypes,
        null_counts=null_counts,
        unique_narratives=unique_narratives,
        duplicate_narratives_count=duplicate_narratives_count,
        class_distribution=class_counts,
        class_ratios=class_ratios,
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def verify_dataset_fingerprint(
    df_or_path: Union[pd.DataFrame, str, Path],
    expected: Union[DatasetFingerprint, str, Path, dict[str, Any]],
    check_file_hash: bool = False,
    raise_on_mismatch: bool = True,
) -> tuple[bool, list[str]]:
    """Verifies that a dataset matches its expected fingerprint.

    Args:
        df_or_path: Loaded DataFrame or path to file.
        expected: DatasetFingerprint instance, dict, or path to saved manifest JSON.
        check_file_hash: If True and file path is provided, also verifies exact raw file SHA-256.
        raise_on_mismatch: If True, raises DatasetFingerprintMismatchError on any failure.

    Returns:
        (is_valid, list_of_mismatch_reasons)
    """
    if isinstance(expected, (str, Path)):
        expected_fp = DatasetFingerprint.load(expected)
    elif isinstance(expected, dict):
        expected_fp = DatasetFingerprint.from_dict(expected)
    else:
        expected_fp = expected

    current_fp = create_dataset_fingerprint(df_or_path)
    mismatches = []

    # Check row counts
    if current_fp.total_rows != expected_fp.total_rows:
        mismatches.append(f"Row count mismatch: expected {expected_fp.total_rows}, got {current_fp.total_rows}")

    # Check content hash
    if current_fp.content_sha256 != expected_fp.content_sha256:
        mismatches.append(
            f"Content SHA-256 mismatch:\n  expected: {expected_fp.content_sha256}\n  got:      {current_fp.content_sha256}"
        )

    # Check file hash if requested and available
    if check_file_hash and expected_fp.file_sha256 and current_fp.file_sha256:
        if current_fp.file_sha256 != expected_fp.file_sha256:
            mismatches.append(
                f"File SHA-256 mismatch:\n  expected: {expected_fp.file_sha256}\n  got:      {current_fp.file_sha256}"
            )

    # Check class distribution
    if current_fp.class_distribution != expected_fp.class_distribution:
        mismatches.append(
            f"Class distribution mismatch:\n  expected: {expected_fp.class_distribution}\n  got:      {current_fp.class_distribution}"
        )

    # Check duplicate count
    if current_fp.duplicate_narratives_count != expected_fp.duplicate_narratives_count:
        mismatches.append(
            f"Duplicate narratives count mismatch: expected {expected_fp.duplicate_narratives_count}, got {current_fp.duplicate_narratives_count}"
        )

    is_valid = len(mismatches) == 0
    if not is_valid and raise_on_mismatch:
        error_msg = "Dataset fingerprint verification failed:\n" + "\n".join(f"- {m}" for m in mismatches)
        raise DatasetFingerprintMismatchError(error_msg)

    return is_valid, mismatches


def create_split_fingerprint(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    label_col: str = LABEL_COL,
    text_col: str = TEXT_COL,
) -> SplitFingerprint:
    """Creates a SplitFingerprint and verifies split integrity invariants."""
    total_count = len(df)
    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)

    # Invariant 1: Total counts must match
    if n_train + n_val + n_test != total_count:
        raise SplitFingerprintMismatchError(
            f"Split sizes sum ({n_train} + {n_val} + {n_test} = {n_train + n_val + n_test}) does not equal dataset size ({total_count})"
        )

    # Invariant 2: Mutually disjoint indices
    s_train, s_val, s_test = set(train_idx), set(val_idx), set(test_idx)
    overlap_train_val = s_train.intersection(s_val)
    overlap_train_test = s_train.intersection(s_test)
    overlap_val_test = s_val.intersection(s_test)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise SplitFingerprintMismatchError(
            f"Overlapping split indices detected! Overlaps: train-val={len(overlap_train_val)}, "
            f"train-test={len(overlap_train_test)}, val-test={len(overlap_val_test)}"
        )

    # Invariant 3: Zero duplicate narrative leakage across splits
    train_texts = set(df.iloc[train_idx][text_col])
    val_texts = set(df.iloc[val_idx][text_col])
    test_texts = set(df.iloc[test_idx][text_col])

    leak_train_val = train_texts.intersection(val_texts)
    leak_train_test = train_texts.intersection(test_texts)
    leak_val_test = val_texts.intersection(test_texts)

    if leak_train_val or leak_train_test or leak_val_test:
        raise SplitFingerprintMismatchError(
            f"Duplicate narratives cross split boundaries! Leaks: train-val={len(leak_train_val)}, "
            f"train-test={len(leak_train_test)}, val-test={len(leak_val_test)}"
        )

    dataset_content_sha256 = compute_dataframe_content_hash(df, text_col=text_col, label_col=label_col)

    train_class_dist = {str(k): int(v) for k, v in df.iloc[train_idx][label_col].value_counts().items()}
    val_class_dist = {str(k): int(v) for k, v in df.iloc[val_idx][label_col].value_counts().items()}
    test_class_dist = {str(k): int(v) for k, v in df.iloc[test_idx][label_col].value_counts().items()}

    return SplitFingerprint(
        dataset_content_sha256=dataset_content_sha256,
        train_indices_sha256=compute_array_sha256(train_idx),
        val_indices_sha256=compute_array_sha256(val_idx),
        test_indices_sha256=compute_array_sha256(test_idx),
        train_count=n_train,
        val_count=n_val,
        test_count=n_test,
        total_count=total_count,
        train_class_distribution=train_class_dist,
        val_class_distribution=val_class_dist,
        test_class_distribution=test_class_dist,
        train_ratio=round(n_train / total_count, 4),
        val_ratio=round(n_val / total_count, 4),
        test_ratio=round(n_test / total_count, 4),
        disjoint_indices_verified=True,
        zero_duplicate_cross_split_verified=True,
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def verify_split_fingerprint(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    expected: Union[SplitFingerprint, str, Path, dict[str, Any]],
    raise_on_mismatch: bool = True,
) -> tuple[bool, list[str]]:
    """Verifies that split arrays and DataFrame match the expected split fingerprint."""
    if isinstance(expected, (str, Path)):
        expected_fp = SplitFingerprint.load(expected)
    elif isinstance(expected, dict):
        expected_fp = SplitFingerprint.from_dict(expected)
    else:
        expected_fp = expected

    current_fp = create_split_fingerprint(df, train_idx, val_idx, test_idx)
    mismatches = []

    if current_fp.dataset_content_sha256 != expected_fp.dataset_content_sha256:
        mismatches.append("Split dataset content hash does not match expected dataset content hash")

    if current_fp.train_indices_sha256 != expected_fp.train_indices_sha256:
        mismatches.append("Train indices SHA-256 mismatch")

    if current_fp.val_indices_sha256 != expected_fp.val_indices_sha256:
        mismatches.append("Validation indices SHA-256 mismatch")

    if current_fp.test_indices_sha256 != expected_fp.test_indices_sha256:
        mismatches.append("Test indices SHA-256 mismatch")

    if current_fp.train_count != expected_fp.train_count:
        mismatches.append(f"Train size mismatch: expected {expected_fp.train_count}, got {current_fp.train_count}")

    if current_fp.val_count != expected_fp.val_count:
        mismatches.append(f"Val size mismatch: expected {expected_fp.val_count}, got {current_fp.val_count}")

    if current_fp.test_count != expected_fp.test_count:
        mismatches.append(f"Test size mismatch: expected {expected_fp.test_count}, got {current_fp.test_count}")

    is_valid = len(mismatches) == 0
    if not is_valid and raise_on_mismatch:
        error_msg = "Split fingerprint verification failed:\n" + "\n".join(f"- {m}" for m in mismatches)
        raise SplitFingerprintMismatchError(error_msg)

    return is_valid, mismatches


@dataclass
class PreprocessingVersion:
    """Explicit record of what preprocessing produced a given experiment's inputs.

    Changes whenever a meaningful preprocessing rule changes - the version_id is a
    human label bumped by hand, not an auto-generated hash, so a diff in
    `project_plan.md`/git history explains *why* it changed alongside *that* it did.
    """
    version_id: str
    dataset_content_sha256: str
    normalization_steps: list[str]
    tokenizer_vocab_size: int
    max_features: int
    oov_token: str
    padding: str
    truncating: str
    max_len_by_experiment: dict[str, int]
    glove_source_path: str
    glove_dim: int
    glove_coverage_pct: float
    distilbert_checkpoint: str
    distilbert_max_len: int
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessingVersion":
        return cls(**data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PreprocessingVersion":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessing version manifest not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
