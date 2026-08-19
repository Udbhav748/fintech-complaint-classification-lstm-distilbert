# CFPB Complaint Classification

5-class financial complaint text classification comparing a recurrent model ladder (LSTM variants) against a fine-tuned transformer (DistilBERT).

---

## Problem

Consumers submit unstructured complaint narratives to the Consumer Financial Protection Bureau (CFPB). The task is to classify each narrative into one of five primary financial product categories:
1. **Checking or savings account**
2. **Credit card**
3. **Debt collection**
4. **Money transfer, virtual currency, or money service**
5. **Student loan**

The goal is not simply to chase a benchmark score, but to isolate the empirical contribution of each architectural, representation, and optimization enhancement across a disciplined ladder.

---

## Dataset

- **Source:** CFPB Consumer Complaint Database (`data/combined_complaints.parquet`).
- **Raw Extract:** 107,992 complaints across the 5 target categories.
- **Deduplication:** Exact duplicate narratives (6,190 boilerplate submissions across 848 distinct texts) are removed prior to splitting, leaving **101,802** unique complaints.
- **Class Balance:** Approximately 1.15:1 ratio post-deduplication (Debt collection: 24,007; Checking/savings: 21,547; Money transfer: 21,437; Credit card: 20,890; Student loan: 20,111).
- **Split:** Stratified 80% train (81,441), 10% validation (10,180), 10% test (10,181), frozen on disk in `data/splits/`.
- **Integrity:** Verified with cryptographic and content SHA-256 hashes (`data/dataset_manifest.json` and `data/splits/split_manifest.json`).

---

## Experimental Design

- **Single Controlled Variable:** Each step in the ladder introduces one isolated change (representation, directionality, regularization, or optimization).
- **Fixed Split:** All models train and evaluate on the exact same frozen splits.
- **Checkpoint Policy:** `ModelCheckpoint(save_best_only=True, monitor='val_macro_f1')` restores best validation weights on every run.
- **Stability References:** Multi-seed evaluation (seeds 42, 123, 456) for M0 baseline and D0 transformer.

---

## Model Ladder

| Model | Architecture | Key Change | Hypothesis |
|---|---|---|---|
| **M0** | Unidirectional LSTM | Random embeddings, `max_len=128`, hidden dim 128 | Baseline stability reference across 3 seeds |
| **M1** | Bidirectional LSTM | Bidirectional recurrent layer | Captures backward context (modest gain expected) |
| **M2** | BiLSTM + Spatial Dropout | Spatial Dropout (0.2) + Dropout (0.3) | Regularization to reduce train/val gap |
| **M3** | BiLSTM + Pretrained GloVe | GloVe-100d pretrained vectors | Accelerates convergence via pretrained semantics |
| **M4** | BiLSTM + GloVe + Optimized | LR schedule + EarlyStopping + `max_len=256` | Efficiency and extended sequence length |
| **D0** | DistilBERT | Fine-tuned `distilbert-base-uncased` | Pretrained contextual transformer benchmark |

---

## Evaluation

- **Primary Metric:** Macro-Averaged F1 (`macro_f1`).
- **Secondary Metrics:** Accuracy, Macro-Precision, Macro-Recall, Per-class F1, Confusion Matrix.
- **Delta Metrics:**
  - $\Delta\text{ vs Previous} = \text{Macro-F1}_{\text{current}} - \text{Macro-F1}_{\text{previous}}$
  - $\Delta\text{ vs M0} = \text{Macro-F1}_{\text{current}} - \text{Macro-F1}_{\text{M0}}$

---

## Results

*Results are recorded to `results/runs.csv` during execution and compiled dynamically via `src.results.generate_comparison_table()`.*

| Model | Configuration | Macro-F1 | Accuracy | Δ vs Previous | Δ vs M0 | Interpretation |
|---|---|---|---|---|---|---|
| **M0** | Unidirectional LSTM baseline, random embeddings | — | — | — | — | Baseline reference |
| **M1** | Bidirectional LSTM | — | — | — | — | Directionality effect |
| **M2** | BiLSTM + Spatial Dropout | — | — | — | — | Regularization effect |
| **M3** | BiLSTM + Pretrained GloVe-100d | — | — | — | — | Pretrained representation |
| **M4** | BiLSTM + GloVe + LR schedule + EarlyStopping | — | — | — | — | Optimization & length bundle |
| **D0** | DistilBERT (fine-tuned transformer) | — | — | — | — | Transformer comparison |

---

## Error Analysis

*(To be populated following final test set evaluation)*
- Hardest class distinctions (e.g. Credit Card vs Checking/Savings dispute narratives).
- Impact of CFPB `XXXX` redaction tokens on tokenization and classification.
- Truncation error analysis for long complaints (>256 words).

---

## Limitations

- **M4 Enhancement Bundling:** Learning rate scheduling, early stopping, and extending `max_len` from 128 to 256 are bundled in M4 due to compute budget. The standalone contribution of `max_len` is not isolated.
- **Redaction Reliance:** CFPB data contains synthetic redaction tokens (`XXXX`) whose distribution varies by product category.
- **Hardware Variation:** Training runtimes reflect CPU / single-GPU environments and are recorded in `results/runs.csv`.

---

## Reproduction

### Environment Setup
```bash
python -m venv .venv
# Activate virtual environment:
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Verification & Split Freezing
```bash
python src/verify_all.py
```

### Running Unit Tests
```bash
python -m unittest discover tests
```

---

## Project Structure

```text
├── configs/
│   └── data_config.json          # Central dataset and tokenization parameters
├── data/
│   ├── combined_complaints.parquet # Master dataset (107,992 rows)
│   ├── dataset_manifest.json     # Cryptographic & content dataset fingerprint
│   ├── splits/                   # Frozen train/val/test split indices (.npy)
│   └── embeddings/               # GloVe pretrained vectors
├── notebooks/
│   ├── 01_eda.ipynb              # Dataset audit, length, and GloVe coverage analysis
│   ├── 02_baselines.ipynb        # M0 baseline LSTM training and stability check
│   ├── 03_ladder.ipynb           # M1–M4 recurrent ladder experiments
│   └── 04_distilbert.ipynb       # D0 DistilBERT fine-tuning and evaluation
├── results/
│   ├── runs.csv                  # Immutable experiment execution record
│   └── metadata/                 # Detailed per-run JSON metadata
├── src/
│   ├── data.py                   # Data loading, deduplication, and labels
│   ├── fingerprint.py            # Dataset & split cryptographic fingerprinting
│   ├── split.py                  # Stratified splitting and invariant validation
│   ├── evaluation.py             # Macro-F1, metrics, and delta calculations
│   ├── checkpoint.py             # Deterministic checkpoints and metric gating
│   ├── config.py                 # Configuration loader and validator
│   ├── logging_utils.py          # Structured lightweight logger
│   ├── reproducibility.py        # Multi-framework seed control and environment capture
│   └── results.py                # runs.csv schema validation and table generator
└── tests/                        # Focused CPU unit test suite
```
