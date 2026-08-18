"""Unit and invariant tests for Phase 5 Simple LSTM Baseline (E0).

Verifies:
  1. Model architecture parameters and forward pass shapes.
  2. Model checkpoint and metadata files exist.
  3. Learning curve and confusion matrix figures exist.
  4. Experiment registry CSV contains valid E0 entry.
  5. Sample predictions are properly formatted with confidence scores.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
import torch

from src.models.lstm import SimpleLSTMClassifier

MODEL_DIR = Path("models/e0_lstm_baseline")
FIGURES_DIR = Path("reports/figures/phase5")
REGISTRY_PATH = Path("reports/experiment_results.csv")
SAMPLE_PREDS_PATH = Path("reports/e0_sample_predictions.json")


def test_model_architecture_and_forward():
    """Verify SimpleLSTMClassifier forward pass shape and parameter counts."""
    vocab_size = 25000
    embedding_dim = 128
    hidden_dim = 64
    num_classes = 5

    model = SimpleLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=1,
        bidirectional=False,
    )

    batch_size = 4
    seq_len = 128
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_lengths = torch.tensor([128, 64, 32, 10])

    logits = model(dummy_input, lengths=dummy_lengths)
    assert logits.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"

    param_dict = model.count_parameters()
    assert param_dict["total_parameters"] == 3249989
    assert param_dict["trainable_parameters"] == 3249989
    assert param_dict["non_trainable_parameters"] == 0


def test_model_artifacts_exist():
    """Verify that E0 saved checkpoint and config files exist."""
    assert (MODEL_DIR / "model.pt").exists(), "model.pt does not exist!"
    assert (MODEL_DIR / "model_config.json").exists(), "model_config.json does not exist!"

    with open(MODEL_DIR / "model_config.json") as f:
        meta = json.load(f)

    assert meta["experiment_id"] == "E0"
    assert meta["architecture"] == "simple_lstm"
    assert meta["test_metrics"]["macro_f1"] > 0.75
    assert meta["test_metrics"]["accuracy"] > 0.75


def test_phase5_figures_exist():
    """Verify that learning curve and confusion matrix plots exist."""
    assert (FIGURES_DIR / "baseline_learning_curves.png").exists()
    assert (FIGURES_DIR / "baseline_confusion_matrix.png").exists()


def test_experiment_registry():
    """Verify that experiment_results.csv contains a valid E0 entry."""
    assert REGISTRY_PATH.exists()
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    e0_records = [r for r in reader if r.get("experiment_id") == "E0"]
    assert len(e0_records) == 1, "E0 not found in experiment registry!"
    e0 = e0_records[0]
    assert float(e0["macro_f1"]) > 0.80
    assert float(e0["accuracy"]) > 0.80
    assert int(e0["epochs"]) == 10


def test_sample_predictions():
    """Verify sample predictions JSON structure."""
    assert SAMPLE_PREDS_PATH.exists()
    with open(SAMPLE_PREDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_id"] == "E0"
    assert len(data["correct_samples"]) > 0
    assert len(data["error_samples"]) > 0
    for s in data["correct_samples"] + data["error_samples"]:
        assert "complaint_id" in s
        assert "true_label" in s
        assert "predicted_label" in s
        assert "confidence" in s
        assert 0.0 <= s["confidence"] <= 1.0
