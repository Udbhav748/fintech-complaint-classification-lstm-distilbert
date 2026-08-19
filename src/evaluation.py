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

from src.data import LABELS, SHORT_LABELS


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

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"y_true length ({len(y_true_arr)}) must equal y_pred length ({len(y_pred_arr)})"
        )

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
