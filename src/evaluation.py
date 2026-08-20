"""Evaluation metrics, confusion matrix, and delta calculation layer.

Provides exact calculations for:
- Macro-F1
- Accuracy
- Macro Precision
- Macro Recall
- Per-class precision, recall, f1, and support
- Confusion matrix
- Delta calculations (vs Previous, vs M0)

All calculations are tested against scikit-learn for mathematical equivalence.
"""

from typing import Any, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data import LABELS


def compute_metrics(
    y_true: Union[np.ndarray, list[int], list[str]],
    y_pred: Union[np.ndarray, list[int], list[str]],
    labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Computes standard classification metrics with exact scikit-learn alignment.

    Args:
        y_true: Ground truth class indices or label strings.
        y_pred: Predicted class indices or label strings.
        labels: List of class labels in canonical order (default: src.data.LABELS).

    Returns:
        Dictionary containing:
        - accuracy (float)
        - macro_f1 (float)
        - macro_precision (float)
        - macro_recall (float)
        - per_class (dict with precision, recall, f1, support per label)
        - confusion_matrix (list of lists, shape [num_classes, num_classes])
    """
    if labels is None:
        labels = LABELS

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError(
            f"y_true/y_pred must be 1-D, got shapes {y_true_arr.shape} and {y_pred_arr.shape}"
        )

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"y_true length ({len(y_true_arr)}) must equal y_pred length ({len(y_pred_arr)})"
        )

    if len(y_true_arr) == 0:
        raise ValueError("y_true/y_pred must not be empty")

    # Fail loudly on out-of-range labels rather than let sklearn silently drop
    # them from a class-restricted average - an out-of-range prediction is a bug
    # upstream (e.g. an argmax over the wrong number of output units), not a
    # class this metric should quietly ignore.
    if np.issubdtype(y_true_arr.dtype, np.integer) or np.issubdtype(y_pred_arr.dtype, np.integer):
        if not (np.issubdtype(y_true_arr.dtype, np.integer) and np.issubdtype(y_pred_arr.dtype, np.integer)):
            raise ValueError(
                f"y_true and y_pred must use the same label type (both integer ids or both "
                f"strings), got dtypes {y_true_arr.dtype} and {y_pred_arr.dtype}"
            )
        num_classes = len(labels)
        out_of_range = np.concatenate(
            [y_true_arr[(y_true_arr < 0) | (y_true_arr >= num_classes)],
             y_pred_arr[(y_pred_arr < 0) | (y_pred_arr >= num_classes)]]
        )
        if out_of_range.size > 0:
            raise ValueError(
                f"Label id(s) {sorted(set(out_of_range.tolist()))} fall outside the valid range "
                f"[0, {num_classes}) for {num_classes} canonical classes"
            )
    else:
        unknown = (set(y_true_arr.tolist()) | set(y_pred_arr.tolist())) - set(labels)
        if unknown:
            raise ValueError(f"Label(s) {sorted(unknown)} are not in the canonical label set {labels}")

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    # Per-class scores
    num_classes = len(labels)
    # Check if inputs are integer indices or string labels
    if np.issubdtype(y_true_arr.dtype, np.integer):
        class_indices = list(range(num_classes))
        prec_per = precision_score(y_true_arr, y_pred_arr, labels=class_indices, average=None, zero_division=0)
        rec_per = recall_score(y_true_arr, y_pred_arr, labels=class_indices, average=None, zero_division=0)
        f1_per = f1_score(y_true_arr, y_pred_arr, labels=class_indices, average=None, zero_division=0)
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=class_indices)
        supports = [int(np.sum(y_true_arr == idx)) for idx in class_indices]
    else:
        prec_per = precision_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
        rec_per = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
        f1_per = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
        supports = [int(np.sum(y_true_arr == lbl)) for lbl in labels]

    per_class_dict = {}
    for i, lbl in enumerate(labels):
        per_class_dict[lbl] = {
            "precision": float(prec_per[i]),
            "recall": float(rec_per[i]),
            "f1": float(f1_per[i]),
            "support": supports[i],
        }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
    }


def calculate_deltas(
    current_f1: float,
    previous_f1: Optional[float] = None,
    baseline_m0_f1: Optional[float] = None,
) -> dict[str, Optional[float]]:
    """Calculates Δ vs Previous and Δ vs M0 unrounded floats.

    Sign convention:
        Positive (+) indicates improvement.
        Negative (-) indicates decline.

    Args:
        current_f1: Macro-F1 of current model.
        previous_f1: Macro-F1 of immediately preceding stage (or None for baseline).
        baseline_m0_f1: Macro-F1 of M0 baseline (or None).

    Returns:
        dict with "delta_vs_previous" and "delta_vs_m0".
    """
    delta_prev = (current_f1 - previous_f1) if previous_f1 is not None else None
    delta_m0 = (current_f1 - baseline_m0_f1) if baseline_m0_f1 is not None else None

    return {
        "delta_vs_previous": delta_prev,
        "delta_vs_m0": delta_m0,
    }


def format_delta(delta: Optional[float], decimals: int = 4) -> str:
    """Formats delta for presentation without altering internal calculation.

    Examples:
        +0.0160, -0.0050, +0.0000, —
    """
    if delta is None:
        return "—"
    sign = "+" if delta > 0 else ("-" if delta < 0 else "+")
    return f"{sign}{abs(delta):.{decimals}f}"


def format_metric(val: Optional[float], std: Optional[float] = None, decimals: int = 4) -> str:
    """Formats metric value with optional standard deviation."""
    if val is None or np.isnan(val):
        return "—"
    if std is not None and not np.isnan(std) and std > 0:
        return f"{val:.{decimals}f} ± {std:.{decimals}f}"
    return f"{val:.{decimals}f}"


def seed_statistics(values: Union[np.ndarray, list[float]], ddof: int = 1) -> dict[str, Any]:
    """Mean/std/n across seed runs - the single place this project computes it.

    `results.generate_comparison_table` calls this rather than recomputing
    `np.std(..., ddof=1)` inline, so M0's and D0's seed spread is calculated the
    same way everywhere.

    `ddof=1` (sample standard deviation, Bessel-corrected) is the project default:
    3 seeds is a sample of the seed population, not the population itself. With
    fewer than 2 values, std is undefined (not zero) and reported as None - a
    single-seed rung (M1-M4) has no seed spread to report, and treating that as
    "std=0" would misrepresent it as a measured stability rather than an absence
    of the measurement. Callers must not apply this to single-seed rungs as if it
    were a real statistic (see `project_plan.md` §9's stability-reference rule).
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        raise ValueError("seed_statistics requires at least one value")
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=ddof)) if n > 1 else None,
        "n": n,
        "ddof": ddof,
    }


def compute_val_macro_f1(
    y_true: Union[np.ndarray, list[int]],
    y_pred: Union[np.ndarray, list[int]],
    labels: Optional[list[str]] = None,
) -> float:
    """The canonical `val_macro_f1` checkpoint metric - call this once per epoch
    on predictions over the FULL validation set, never as a running per-batch
    average.

    Macro-F1 is not decomposable across batches: precision and recall for each
    class depend on totals (true positives, false positives, false negatives)
    accumulated over the whole set, and F1 is a nonlinear (harmonic-mean)
    combination of those. Averaging per-batch macro-F1 scores gives a different,
    generally biased number - worse for small batches, and the bias does not
    cancel out over an epoch. `tests/test_evaluation.py` demonstrates this
    concretely with a constructed example where the two disagree.

    This is a thin wrapper around `compute_metrics`'s `macro_f1` (itself verified
    against `sklearn.metrics.f1_score(average="macro")`), kept as its own
    function so the model factory (Task 5) has one unambiguous, tested call site
    to use inside a Keras `on_epoch_end` callback: predict on the full validation
    set, call this, compare against the best checkpoint's value.
    """
    return compute_metrics(y_true, y_pred, labels=labels)["macro_f1"]
