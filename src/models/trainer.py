"""Training and Evaluation engine for PyTorch models.

Provides deterministic training loop, metric evaluation (Accuracy, Macro-F1, Precision,
Recall, Confusion Matrix), learning curve plotting, and experiment registry logging.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Set deterministic random seed = %d", seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train model for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths=lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += len(labels)

    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a dataloader and return loss, accuracy, macro-F1, y_true, y_pred, y_prob."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, lengths=lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    total_samples = len(all_labels)
    avg_loss = total_loss / total_samples
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.vstack(all_probs)

    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    return avg_loss, acc, macro_f1, y_true, y_pred, y_prob


def plot_learning_curves(
    history: dict[str, list[float]],
    output_path: Path | str = "reports/figures/phase5/baseline_learning_curves.png",
) -> Path:
    """Plot training and validation loss & accuracy curves over epochs."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], "o-", color="#2b5c8f", label="Train Loss", linewidth=2.0)
    ax1.plot(epochs, history["val_loss"], "s--", color="#c0392b", label="Val Loss", linewidth=2.0)
    ax1.set_title("E0 Baseline: Loss vs Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy Curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", color="#2b5c8f", label="Train Acc", linewidth=2.0)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", color="#3e8e7e", label="Val Acc", linewidth=2.0)
    ax2.set_title("E0 Baseline: Accuracy vs Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved learning curves plot: %s", out)
    return out


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path | str = "reports/figures/phase5/baseline_confusion_matrix.png",
) -> Path:
    """Plot formatted confusion matrix with raw counts and row-normalized percentages."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    # Compute row-normalized proportions
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Shorten class names for tick labels
    display_names = [c.replace(", ", ",\n") for c in class_names]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=display_names,
        yticklabels=class_names,
        title="E0 Simple LSTM Baseline: Test Confusion Matrix (N = 16,197)",
        ylabel="True Product Label",
        xlabel="Predicted Product Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=8.5)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            ax.text(
                j,
                i,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if count > thresh else "black",
                fontsize=8.5,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix plot: %s", out)
    return out


def update_experiment_registry(
    record: dict[str, Any],
    registry_path: Path | str = "reports/experiment_results.csv",
) -> None:
    """Add or update an experiment record in the experiment registry CSV."""
    reg_path = Path(registry_path)
    reg_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id",
        "model",
        "dataset_version",
        "embedding_type",
        "vocabulary_size",
        "embedding_dim",
        "lstm_units",
        "max_length",
        "batch_size",
        "learning_rate",
        "epochs",
        "seed",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "accuracy",
        "training_time_seconds",
        "total_parameters",
        "trainable_parameters",
        "notes",
    ]

    existing_rows: list[dict[str, Any]] = []
    if reg_path.exists():
        with open(reg_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    # Filter out existing record with same experiment_id if updating
    filtered_rows = [r for r in existing_rows if r.get("experiment_id") != record.get("experiment_id")]
    filtered_rows.append(record)

    with open(reg_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in filtered_rows:
            # Format float metrics
            row_dict = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, float):
                    row_dict[k] = f"{v:.4f}"
                else:
                    row_dict[k] = str(v)
            writer.writerow(row_dict)

    logger.info("Updated experiment registry: %s with experiment %s", reg_path, record.get("experiment_id"))
