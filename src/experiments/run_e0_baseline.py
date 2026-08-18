"""Phase 5: Run Experiment E0 (Simple LSTM Baseline).

Executes training of the simple single-layer unidirectional LSTM on the 75,598-sample
training split, evaluates on validation during training, evaluates on test set after training,
saves model artifacts, generates plots, and writes the baseline report.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure cache directories stay on D:
os.environ["HF_HOME"] = str(Path("D:/AI-Models/huggingface").absolute())
os.environ["TRANSFORMERS_CACHE"] = str(Path("D:/AI-Models/huggingface/transformers").absolute())
os.environ["TORCH_HOME"] = str(Path("D:/AI-Models/torch").absolute())

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from src.models.dataset import create_dataloaders, load_vocab_and_mappings
from src.models.lstm import SimpleLSTMClassifier
from src.models.trainer import (
    evaluate,
    plot_confusion_matrix,
    plot_learning_curves,
    set_seed,
    train_epoch,
    update_experiment_registry,
)

logger = logging.getLogger("e0_baseline")


def setup_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: Path | str = "configs/experiments/e0_lstm_baseline.yaml") -> dict[str, Any]:
    """Load YAML experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_e0_experiment(config_path: Path | str = "configs/experiments/e0_lstm_baseline.yaml") -> dict[str, Any]:
    """Execute the full E0 Simple LSTM Baseline training and evaluation pipeline."""
    setup_logging()
    logger.info("Starting Phase 5: Experiment E0 (Simple LSTM Baseline)...")

    cfg = load_config(config_path)

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    # Setup directories on D:
    model_dir = Path(cfg["output"]["model_dir"])
    figures_dir = Path(cfg["output"]["figures_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Vocabulary and Label Mappings
    logger.info("=== Loading Vocabulary and Dataset ===")
    vocab, name_to_id, id_to_name = load_vocab_and_mappings(
        vocab_path=cfg["data"]["vocab_path"],
        label_mapping_path=cfg["data"]["label_mapping_path"],
    )

    class_names = [id_to_name[i] for i in range(len(id_to_name))]

    # 2. Create DataLoaders
    max_length = int(cfg["data"]["max_length"])
    batch_size = int(cfg["training"]["batch_size"])

    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=cfg["data"]["train_path"],
        val_path=cfg["data"]["val_path"],
        test_path=cfg["data"]["test_path"],
        vocab=vocab,
        name_to_id=name_to_id,
        max_length=max_length,
        batch_size=batch_size,
    )

    # 3. Instantiate Simple LSTM Model
    device = torch.device(cfg["training"].get("device", "cpu"))
    logger.info("Instantiating SimpleLSTMClassifier on device: %s", device)

    model = SimpleLSTMClassifier(
        vocab_size=int(cfg["model"]["vocab_size"]),
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_classes=int(cfg["data"]["num_classes"]),
        num_layers=int(cfg["model"]["num_layers"]),
        bidirectional=bool(cfg["model"]["bidirectional"]),
        dropout=float(cfg["model"]["dropout"]),
        pad_idx=vocab.pad_idx,
    ).to(device)

    param_counts = model.count_parameters()
    logger.info("Model Parameter Counts: %s", param_counts)

    # 4. Setup Optimizer & Loss Function
    learning_rate = float(cfg["training"]["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    epochs = int(cfg["training"]["epochs"])
    logger.info("Starting training for %d epochs...", epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_f1, 4))

        logger.info(
            "Epoch %2d/%2d [%.1fs] - Train Loss: %.4f, Train Acc: %.2f%% | Val Loss: %.4f, Val Acc: %.2f%%, Val Macro-F1: %.4f",
            epoch,
            epochs,
            epoch_time,
            train_loss,
            train_acc * 100.0,
            val_loss,
            val_acc * 100.0,
            val_f1,
        )

    total_training_time = time.time() - start_time
    logger.info("Training finished in %.2f seconds (%.2f minutes)", total_training_time, total_training_time / 60.0)

    # 6. Evaluation on Held-Out Test Set
    logger.info("=== Evaluating on Held-Out Test Set (N = 16,197) ===")
    test_loss, test_acc, test_macro_f1, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device)

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average="macro")
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(y_true, y_pred, average=None)

    logger.info(
        "TEST RESULTS -> Macro-F1: %.4f | Accuracy: %.2f%% | Precision: %.4f | Recall: %.4f",
        test_macro_f1,
        test_acc * 100.0,
        precision,
        recall,
    )

    per_class_metrics = {}
    for i, cls_name in enumerate(class_names):
        per_class_metrics[cls_name] = {
            "class_id": i,
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_supp[i]),
        }
        logger.info(
            "  Class %d (%s) -> Precision: %.4f, Recall: %.4f, F1: %.4f, Support: %d",
            i,
            cls_name,
            per_class_p[i],
            per_class_r[i],
            per_class_f1[i],
            per_class_supp[i],
        )

    # 7. Generate Plots
    logger.info("=== Generating Figures ===")
    learning_curves_path = plot_learning_curves(history, output_path=figures_dir / "baseline_learning_curves.png")

    cm = confusion_matrix(y_true, y_pred)
    cm_path = plot_confusion_matrix(cm, class_names=class_names, output_path=figures_dir / "baseline_confusion_matrix.png")

    # 8. Save Model and Artifacts
    logger.info("=== Saving Model Checkpoint ===")
    model_save_path = model_dir / "model.pt"
    torch.save(model.state_dict(), model_save_path)

    model_metadata = {
        "experiment_id": cfg["experiment_id"],
        "architecture": cfg["model"]["architecture"],
        "dataset_version": cfg["data"]["dataset_version"],
        "vocab_size": cfg["model"]["vocab_size"],
        "embedding_dim": cfg["model"]["embedding_dim"],
        "hidden_dim": cfg["model"]["hidden_dim"],
        "max_length": max_length,
        "num_classes": len(class_names),
        "class_names": class_names,
        "seed": seed,
        "parameter_counts": param_counts,
        "training_time_seconds": round(total_training_time, 2),
        "history": history,
        "test_metrics": {
            "macro_f1": round(test_macro_f1, 4),
            "accuracy": round(test_acc, 4),
            "macro_precision": round(precision, 4),
            "macro_recall": round(recall, 4),
            "per_class": per_class_metrics,
        },
    }

    with open(model_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    # 9. Save Sample Predictions for Error Analysis
    logger.info("=== Generating Sample Predictions ===")
    test_df = pd.read_parquet(cfg["data"]["test_path"]) if str(cfg["data"]["test_path"]).endswith(".parquet") else pd.read_csv(cfg["data"]["test_path"])
    texts = test_df["narrative"].tolist()

    correct_samples: list[dict[str, Any]] = []
    error_samples: list[dict[str, Any]] = []

    for idx in range(len(y_true)):
        true_lbl = class_names[y_true[idx]]
        pred_lbl = class_names[y_pred[idx]]
        prob = float(np.max(y_prob[idx]))
        snippet = texts[idx][:200].replace("\n", " ")

        item = {
            "complaint_id": str(test_df["complaint_id"].iloc[idx]),
            "true_label": true_lbl,
            "predicted_label": pred_lbl,
            "confidence": round(prob, 4),
            "narrative_snippet": snippet,
        }

        if y_true[idx] == y_pred[idx]:
            if len(correct_samples) < 15:
                correct_samples.append(item)
        else:
            if len(error_samples) < 25:
                error_samples.append(item)

    sample_preds_path = Path(cfg["output"]["sample_predictions_path"])
    with open(sample_preds_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": "E0",
                "total_test_samples": len(y_true),
                "correct_samples": correct_samples,
                "error_samples": error_samples,
            },
            f,
            indent=2,
        )

    # 10. Update Experiment Registry
    registry_record = {
        "experiment_id": "E0",
        "model": "Simple LSTM",
        "dataset_version": cfg["data"]["dataset_version"],
        "embedding_type": "Random Trainable",
        "vocabulary_size": cfg["model"]["vocab_size"],
        "embedding_dim": cfg["model"]["embedding_dim"],
        "lstm_units": cfg["model"]["hidden_dim"],
        "max_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed,
        "macro_f1": test_macro_f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "accuracy": test_acc,
        "training_time_seconds": total_training_time,
        "total_parameters": param_counts["total_parameters"],
        "trainable_parameters": param_counts["trainable_parameters"],
        "notes": "Reference baseline E0: single-layer unidirectional LSTM with random trainable embeddings, no dropout, no class weights.",
    }
    update_experiment_registry(registry_record, registry_path=cfg["output"]["experiment_registry_path"])

    logger.info("Phase 5 E0 experiment completed successfully!")
    return {
        "metadata": model_metadata,
        "history": history,
        "test_metrics": model_metadata["test_metrics"],
        "confusion_matrix": cm.tolist(),
        "training_time": total_training_time,
        "param_counts": param_counts,
    }


if __name__ == "__main__":
    run_e0_experiment()
