"""Script to generate and execute notebooks/02_baseline_lstm.ipynb.

Produces the comprehensive baseline experiment notebook documenting the E0 Simple LSTM model,
hyperparameters, training dynamics, test set evaluation, confusion matrix, and learning curves.
"""

from __future__ import annotations

import json
from pathlib import Path
import nbformat as nbf


def create_baseline_notebook(output_path: Path | str = "notebooks/02_baseline_lstm.ipynb") -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    # Title cell
    cells.append(nbf.v4.new_markdown_cell("""# FinTech Complaint Classification: Simple LSTM Baseline (E0)
**Phase 5 — Experiment E0: Simple Single-Layer LSTM Baseline**
- **Corpus Size:** 107,992 verified complaint records (75,598 train, 16,197 val, 16,197 test)
- **Objective:** Establish the un-enhanced recurrent reference performance for controlled comparisons in Test-6.
- **Model:** Single-layer unidirectional LSTM with randomly initialized trainable embeddings (embedding_dim=128, hidden_dim=64).
"""))

    # Section 1: Setup & Environment
    cells.append(nbf.v4.new_markdown_cell("""## 1. Environment & Configuration
Load the experiment configuration from `configs/experiments/e0_lstm_baseline.yaml` and inspect model parameters.
"""))
    cells.append(nbf.v4.new_code_cell("""import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Load experiment configuration
config_path = Path("../configs/experiments/e0_lstm_baseline.yaml")
if not config_path.exists():
    config_path = Path("configs/experiments/e0_lstm_baseline.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print("=== E0 Experiment Configuration ===")
print(json.dumps(config, indent=2))
"""))

    # Section 2: Model Architecture
    cells.append(nbf.v4.new_markdown_cell("""## 2. Model Architecture & Parameter Counts
Review the simple single-layer unidirectional LSTM architecture.
"""))
    cells.append(nbf.v4.new_code_cell("""model_config_path = Path("../models/e0_lstm_baseline/model_config.json")
if not model_config_path.exists():
    model_config_path = Path("models/e0_lstm_baseline/model_config.json")

with open(model_config_path, "r") as f:
    model_meta = json.load(f)

print(f"Architecture: {model_meta['architecture']}")
print(f"Vocabulary Size: {model_meta['vocab_size']:,}")
print(f"Embedding Dimension: {model_meta['embedding_dim']}")
print(f"LSTM Units (Hidden Dim): {model_meta['hidden_dim']}")
print(f"Sequence Length (max_length): {model_meta['max_length']}")
print(f"Total Parameters: {model_meta['parameter_counts']['total_parameters']:,}")
print(f"Trainable Parameters: {model_meta['parameter_counts']['trainable_parameters']:,}")
"""))

    # Section 3: Training History & Learning Curves
    cells.append(nbf.v4.new_markdown_cell("""## 3. Training Dynamics & Learning Curves
Inspect the 10-epoch training loss, validation loss, and validation Macro-F1 progression.
"""))
    cells.append(nbf.v4.new_code_cell("""history = model_meta["history"]
history_df = pd.DataFrame({
    "Epoch": range(1, len(history["train_loss"]) + 1),
    "Train Loss": history["train_loss"],
    "Val Loss": history["val_loss"],
    "Train Acc (%)": [a * 100 for a in history["train_acc"]],
    "Val Acc (%)": [a * 100 for a in history["val_acc"]],
    "Val Macro-F1": history["val_macro_f1"]
})
display(history_df)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
epochs = history_df["Epoch"]

# Loss plot
ax1.plot(epochs, history_df["Train Loss"], "o-", color="#2b5c8f", label="Train Loss", linewidth=2.0)
ax1.plot(epochs, history_df["Val Loss"], "s--", color="#c0392b", label="Val Loss", linewidth=2.0)
ax1.set_title("E0 Baseline: Training & Validation Loss", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.legend()

# Accuracy plot
ax2.plot(epochs, history_df["Train Acc (%)"], "o-", color="#2b5c8f", label="Train Acc", linewidth=2.0)
ax2.plot(epochs, history_df["Val Acc (%)"], "s--", color="#3e8e7e", label="Val Acc", linewidth=2.0)
ax2.set_title("E0 Baseline: Training & Validation Accuracy", fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
plt.tight_layout()
plt.show()
"""))
    cells.append(nbf.v4.new_markdown_cell("""### Training Dynamics Observation
- **Convergence:** The simple LSTM converges rapidly, achieving >82% validation accuracy by Epoch 2.
- **Overfitting Behavior:** Validation loss reaches its minimum at **Epoch 4 (0.4175)** and gradually drifts upward to 0.5361 by Epoch 10, while training loss drops to 0.1610 (94.65% train accuracy).
- **Implication:** The unregularized baseline exhibits mild overfitting due to the absence of dropout or weight decay, providing clear motivation for subsequent regularization and architectural enhancements.
"""))

    # Section 4: Test Set Evaluation Metrics
    cells.append(nbf.v4.new_markdown_cell("""## 4. Test Set Evaluation Metrics
Evaluate performance on the held-out test partition (N = 16,197 records).
"""))
    cells.append(nbf.v4.new_code_cell("""test_metrics = model_meta["test_metrics"]
print(f"=== E0 Simple LSTM Test Set Performance (N = 16,197) ===")
print(f"Macro-F1 (Primary Metric): {test_metrics['macro_f1']:.4f}")
print(f"Accuracy:                 {test_metrics['accuracy']*100:.2f}%")
print(f"Macro Precision:          {test_metrics['macro_precision']:.4f}")
print(f"Macro Recall:             {test_metrics['macro_recall']:.4f}")

per_class_df = pd.DataFrame(test_metrics["per_class"]).T
per_class_df.index.name = "Product Class"
display(per_class_df[["class_id", "precision", "recall", "f1_score", "support"]])
"""))

    # Section 5: Confusion Matrix
    cells.append(nbf.v4.new_markdown_cell("""## 5. Confusion Matrix Analysis
Inspect class-by-class misclassifications on the held-out test set.
"""))
    cells.append(nbf.v4.new_code_cell("""# Display precomputed confusion matrix figure
from IPython.display import Image, display

cm_img_path = Path("../reports/figures/phase5/baseline_confusion_matrix.png")
if not cm_img_path.exists():
    cm_img_path = Path("reports/figures/phase5/baseline_confusion_matrix.png")

if cm_img_path.exists():
    display(Image(filename=str(cm_img_path)))
"""))
    cells.append(nbf.v4.new_markdown_cell("""### Confusion Matrix Observations
1. **High-Performing Isolated Classes:**
   - `Student loan` achieves **0.9551 F1** (94.9% recall, 96.1% precision).
   - `Debt collection` achieves **0.9237 F1** (93.5% recall, 91.3% precision).
   - Both categories feature distinct domain vocabulary (`mohela`, `pslf`, `fdcpa`, `validation`).
2. **Primary Misclassification Axis:**
   - `Checking or savings account` (0.7284 F1) and `Money transfer, virtual currency, or money service` (0.7765 F1) exhibit mutual confusion.
   - 459 Checking/Savings complaints were predicted as Money Transfer, and 378 Money Transfer complaints were predicted as Checking/Savings.
   - Both products involve transaction disputes, wire transfers, unauthorized debits, and banking terminology.
"""))

    # Section 6: Sample Predictions Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 6. Sample Predictions & Error Analysis
Review representative correct and incorrect predictions from the test set.
"""))
    cells.append(nbf.v4.new_code_cell("""samples_path = Path("../reports/e0_sample_predictions.json")
if not samples_path.exists():
    samples_path = Path("reports/e0_sample_predictions.json")

with open(samples_path, "r") as f:
    samples_data = json.load(f)

print("=== Correct Predictions (Sample of 3) ===")
for s in samples_data["correct_samples"][:3]:
    print(f"Complaint ID: {s['complaint_id']} | True: {s['true_label']} | Pred: {s['predicted_label']} (Conf: {s['confidence']:.2f})")
    print(f"  Snippet: \\"{s['narrative_snippet']}...\\"\\n")

print("=== Incorrect Predictions (Sample of 3) ===")
for s in samples_data["error_samples"][:3]:
    print(f"Complaint ID: {s['complaint_id']} | True: {s['true_label']} | Pred: {s['predicted_label']} (Conf: {s['confidence']:.2f})")
    print(f"  Snippet: \\"{s['narrative_snippet']}...\\"\\n")
"""))

    # Section 7: Experiment Registry Status
    cells.append(nbf.v4.new_markdown_cell("""## 7. Experiment Registry
Inspect the updated project experiment registry in `reports/experiment_results.csv`.
"""))
    cells.append(nbf.v4.new_code_cell("""reg_path = Path("../reports/experiment_results.csv")
if not reg_path.exists():
    reg_path = Path("reports/experiment_results.csv")

reg_df = pd.read_csv(reg_path)
display(reg_df)
"""))

    # Section 8: Conclusion
    cells.append(nbf.v4.new_markdown_cell("""## 8. Baseline Conclusion
This establishes **E0** as the reference baseline performance for subsequent controlled enhancement experiments in Test-6.

- **Baseline Macro-F1:** **0.8435**
- **Baseline Accuracy:** **84.45%**
- **Next Controlled Step:** Phase 6 (E1 — Bidirectional LSTM) will test whether processing sequences bidirectionally improves representation quality over this unidirectional baseline.
"""))

    nb.cells = cells
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated notebook: {out_path}")


if __name__ == "__main__":
    create_baseline_notebook()
