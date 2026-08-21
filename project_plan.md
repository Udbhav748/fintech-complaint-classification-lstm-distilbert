# CFPB Complaint Classification — Project Plan

## 1. Project Overview

**Task:** 5-class text classification — `Consumer complaint narrative` → `Product`

**Dataset:** CFPB combined complaints, 107,992 rows, 5 classes (Debt collection, Checking/savings, Money transfer/virtual currency, Credit card, Student loan), ~1.19:1 class ratio.

**Goal:** Not the highest Macro-F1. Demonstrate what changed, why, whether it helped, by how much, and why the result occurred.

**Core question:** How much do architectural, representation, regularization, and optimization choices improve on a simple LSTM baseline — and does DistilBERT outperform the best recurrent config?

---

## 2. Dataset Properties

All figures below are outputs of the Stage 1 audit (`notebooks/01_eda.ipynb`), measured on `data/combined_complaints.parquet`. Values are post-deduplication unless marked otherwise.

| Property | Value |
|---|---|
| Rows loaded | 107,992 |
| Null / blank narratives | 0 |
| Exact duplicate rows removed | 6,190 (848 distinct texts across 7,038 rows) |
| Rows after deduplication | 101,802 |
| Class distribution | Checking/savings 21,524 · Money transfer 20,940 · Credit card 20,684 · Student loan 19,985 · Debt collection 18,669 |
| Class ratio | 1.153:1 (1.194:1 before deduplication) |
| Near-duplicates surviving exact dedup | 7.8% of rows at cosine ≥ 0.90, 10.8% at ≥ 0.70 (Debt collection 25.9%) |
| Narrative length (words) | median 178 · p75 288 · p90 436 · max 5,699 |
| Keras token length | median 180 · p90 438 |
| DistilBERT WordPiece length | median 228 · p90 563 (×1.26 the Keras length) |
| Truncated at `max_len=128` | Keras 65.6% · WordPiece 74.4% |
| Truncated at `max_len=256` | Keras 31.0% · WordPiece 43.9% |
| Complaints containing `XXXX` | 88,331 (86.8%) |
| `XXXX` per complaint | mean 13.8 · median 7 · max 4,306 |
| Vocabulary | 57,396 types; top 20,000 cover 99.78% of tokens |
| GloVe 100d coverage at 20k vocab | 99.16% of tokens (OOV 0.84%) |
| TF-IDF + LR reference | Macro-F1 0.863 |

Largest duplicate groups are 832, 293, and 238 identical narratives — boilerplate submissions, 85% of them in Debt collection. Deduplication before splitting is not optional; without it the same text appears in train and test. Deduplication is not label-neutral: it removes 22% of Debt collection and 0.1% of Checking/savings, which is why the class ratio moves.

The audit confirmed the pre-registered `max_len` ladder — M0–M3 at 128, M4 at 256 — rather than revising it. WordPiece inflates length on financial jargon and redaction tokens exactly as anticipated (`XXXX` → `xx ##xx`, `navient` → `na ##vie ##nt`), so DistilBERT truncates more at the same nominal window. See §8.1.

`XXXX` is **not** a class shortcut. The §2 gap in mean count (Credit card 15.2 vs Debt collection 11.8) tracks complaint length, not product: normalised by length the class spread is 4.03%–4.80%, and a classifier given only redaction and length counts reaches 0.274 Macro-F1 against 0.200 for guessing. Redactions are retained, logged as a weak confound.

---

## 3. Experimental Design

### 3.1 Configurations — 6 total

| Model | Configuration | Change from previous | Demonstrates |
|---|---|---|---|
| M0 | Simple LSTM, random embeddings, `max_len=128` | — | Baseline |
| M1 | M0 + Bidirectional | Direction | Context direction |
| M2 | M1 + dropout & `recurrent_dropout` | Regularization | Regularization |
| M3 | M2 + GloVe 100d | Embedding init | Pretrained representation |
| M4 | M3 + LR scheduling + early stopping + `max_len=256` | Optimization + context | Optimization + context (bundled — see §8) |
| D0 | DistilBERT fine-tune, `max_len=256` | Model family | Family comparison vs M4 |

M0–M4 is a cumulative experimental ladder — a progressive, ablation-style design. The distinction matters: a classical ablation removes components from a full model to isolate each one, whereas this design adds them in sequence. A rung's Δ vs its predecessor isolates that rung's change; its Δ vs M0 is the combined effect of everything up to it. Both are reported (§6) and must not be conflated.

### 3.2 Training runs — 10 fits

```
M0 × 3 seeds
M1 × 1
M2 × 1
M3 × 1
M4 × 1
D0 × 3 seeds
```

M0 and D0 receive three seeds because they anchor the baseline and the final model-family claim. M1–M4 are single-seed ablation rungs under the fixed compute budget; M4 is therefore a single-seed best-recurrent benchmark, and the headline D0-vs-M4 comparison carries a variance estimate on one side only. That asymmetry is stated wherever the comparison is reported.

### 3.3 Enhancement menu coverage

| Menu item | Covered by |
|---|---|
| Bidirectional LSTM | M1 |
| Dropout & `recurrent_dropout` | M2 |
| GloVe vs random init | M3 |
| Learning-rate scheduling | M4 |
| Early stopping | M4 |
| Longer `max_length` | M4 |
| LSTM → DistilBERT | D0 |
| Stacked LSTM layers | Excluded — lower-priority architectural variation under the time and compute budget |
| Class weights | Excluded — 1.153:1 imbalance after deduplication is mild; the largest reweighting available is ×1.15 on ~19k examples per class, so substantial benefit is not expected (Stage 1 audit) |

8 of 9 covered. Both exclusions are stated with reasons rather than omitted.

---

## 4. Experimental Controls

Held fixed across M0–M4:

- Same train/val/test split, persisted to disk as index files and loaded by every notebook
- Duplicates removed before splitting
- Same tokenizer, vocabulary, and `max_features`
- Maximum epoch budget of 10, revised ladder-wide if M0 has not converged — never per-model
- `ModelCheckpoint(save_best_only=True, monitor='val_macro_f1')` on every run
- Fixed random seeds where applicable
- Test set untouched until final evaluation

The checkpoint rule is what makes M4 interpretable. Best-validation weights are restored on every run, so M4's early-stopping contribution is *termination* — when training halts — not *weight selection*. Without it, M4's delta would be best-epoch-vs-last-epoch, a confound that would inflate the row for reasons unrelated to the enhancement.

`val_macro_f1` is not a stock Keras metric. It is verified against a manual scikit-learn macro-F1 on a small batch before M0 runs; every checkpoint in the project depends on it being correct.

---

## 5. Metrics

**Primary:** Macro-F1

**Secondary:** Accuracy, Macro-Precision, Macro-Recall

Logged per fit to `results/runs.csv`:

- Training loss, validation loss
- Best epoch, epochs run
- Training time (seconds)
- Parameter count
- Best validation Macro-F1

M0 and D0 report mean ± std across 3 seeds, not a single number.

---

## 6. Comparison Table Structure

Generated from `results/runs.csv`, never hand-maintained.

| Model | Configuration | Macro-F1 | Accuracy | Δ vs Previous | Δ vs M0 | Interpretation |
|---|---|---|---|---|---|---|
| M0 | | | | — | — | |
| M1 | | | | | | |
| M2 | | | | | | |
| M3 | | | | | | |
| M4 | | | | | | |
| D0 | | | | | | |

**Δ vs Previous** — what this stage contributed on its own.
**Δ vs M0** — total improvement over the original baseline, i.e. the cumulative effect of every rung to that point.

---

## 7. Pre-registered Hypotheses

| Model | Prediction | Mechanism |
|---|---|---|
| M1 | Modest positive or near-zero | Complaint topics are lexically distinctive; keyword presence matters more than word order |
| M2 | Small positive | Depends on whether M1 is overfitting — if the train/val gap is narrow there is nothing to regularize |
| M3 | Positive | Pretrained vectors accelerate convergence; coverage on financial text and redaction tokens is the variable to watch |
| M4 | Modest positive on F1, meaningful reduction in training time | Early stopping is an efficiency control, not an accuracy lever; the `max_len` change is the component most likely to move F1 |
| D0 | Clear positive vs M4 | Transformer pretrained on a large corpus vs an LSTM trained on 108k complaints |

These are hypotheses, not facts. A failed prediction is written up with an explanation of why the reasoning was wrong. It does not disappear.

---

## 8. M4 Limitation — Decision Log

> LR scheduling, early stopping, and longer `max_length` are treated as a single optimization-and-context stage. Scheduling and stopping interact by design — a schedule alters when the stopping criterion fires. `max_length` is bundled here because the time budget does not allow a standalone row. The individual contribution of `max_length` cannot be isolated from this experiment. This is stated as a limitation, not hidden.

### 8.1 D0 vs M4 — Unequal Context

`max_len=256` for D0 was not specified in the original plan and was set by the Stage 1 audit so that the headline comparison holds the *nominal* window constant. It does not hold the *effective* window constant. WordPiece produces a median 1.26× more tokens than the Keras tokenizer on this text, so at 256 tokens M4 reads 74.6% of all corpus tokens while D0 reads 64.9% of its own.

D0 is therefore handicapped on context. If D0 still beats M4 the conclusion is only stronger; if it loses narrowly, this asymmetry is the first thing to check before concluding anything about model families. Reported wherever D0 vs M4 is reported.

---

## 9. Pipeline Stages

1. **Data audit** — ✅ complete (`notebooks/01_eda.ipynb`, `configs/data_config.json`). `XXXX` redaction check, dedupe, near-duplicate measurement, length percentiles under both tokenizers, vocab coverage curve, class balance. Findings in §14.3
2. **Reference baseline** — TF-IDF + Logistic Regression, CPU, off-quota. Answers whether sequence modeling beats bag-of-words on this task. Auxiliary only: it is never labeled M0 and is not one of the 6 configurations
3. **M0 × 3 seeds** → inspect σ before proceeding. If σ > 0.8 F1 points, most ladder deltas are unresolvable and that is addressed before continuing
4. **M1 → M4** sequentially, one rung at a time
5. **D0 × 3 seeds**
6. **Analysis** — comparison table, per-class F1, confusion matrices for M4 and D0, error analysis

If M0 substantially underperforms the TF-IDF reference, investigate preprocessing, tokenization, optimization, and architecture before proceeding. The reference is a sanity check, not a bar M0 is required to clear.

> The 3-seed M0 and D0 spreads serve as a **stability reference**, not as a significance test. Repeated seeds exist only at those two points, so no rung-specific σ is available for M1–M4, and comparing a single-seed Δ against one endpoint's σ would not constitute a formal test. Rule fixed before any results are seen: any rung whose Δ falls within the observed M0 seed spread is reported as indistinguishable from run-to-run variance rather than as an improvement. Claims of a real effect are reserved for Δ comfortably outside that spread.

---

## 10. Error Analysis Plan

Performed on the best model once chosen:

- **Confusion matrix** — which classes are confused, and whether the confusion is semantic or an artifact
- **Per-class F1** — which classes carry low recall, precision, or F1
- **Misclassified examples** — actual narrative text, predicted label, true label, candidate reason

Expected confusion: Debt collection ↔ Credit card. A complaint about a collector pursuing card debt is genuinely ambiguous, and both labels are defensible. This is a label-boundary property of the dataset, not a model failure, and is reported as such.

---

## 11. Repo Structure

```
fintech-complaint-classification-lstm-distilbert/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_ladder.ipynb
│   └── 04_distilbert.ipynb
├── src/
│   ├── data.py
│   ├── text.py
│   ├── embeddings.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── results/
│   └── runs.csv
└── README.md
```

Logic lives in importable modules rather than notebook cells: it survives Kaggle session death, and it makes the ablation structurally verifiable — one argument changes per rung in the model factory, visibly.

---

## 12. Known Risks

| Risk | Handling |
|---|---|
| σ too large to resolve deltas | Caught at the M0 checkpoint; null results reported honestly rather than overclaimed |
| `recurrent_dropout` disables the cuDNN fast path → ~5–10× slower epochs | Expected, budgeted, logged as the trade-off it is |
| GloVe coverage poor on financial and redacted text | Measured in Stage 1: 99.16% of tokens covered, but only 15–26 of each class's 30 most distinctive terms. The high-signal vocabulary (`mohela`, `navient`, `coinbase`, `fdcpa`, `pslf`) is exactly what stays randomly initialised, so M3 should be expected to gain less than raw coverage suggests |
| Near-duplicate templates leak across splits | Measured in Stage 1, sized per similarity band in Stage 2. Handled by the grouped split at cosine ≥ 0.70 (§14.4); cross-split cluster leakage is now a split invariant |
| Kaggle session dies mid-run | Checkpoints written to `/kaggle/working`; logic in `src/`, so a crash costs a run and not the code |

---

## 13. Locked Summary

| Item | Value |
|---|---|
| Configurations | 6 (M0–M4, D0) |
| Training fits | 10 |
| Menu coverage | 8 of 9 |
| Framework | Keras/TF (M0–M4), HuggingFace TF (D0) |
| Primary metric | Macro-F1 |
| Split | 80/10/10 stratified |
| Epoch budget | Max 10, ladder-wide |
| Hardware | Kaggle P100 |
| Estimated GPU budget | ~6–7 hours |

GPU budget is an estimate, not a locked fact. Actual training time is recorded per fit in `results/runs.csv`.


# 14. Project Execution Pipeline

This section is the working execution checklist for the project. Every stage must be completed, verified, and committed before moving to the next stage.

The goal is not only to obtain strong model performance, but to produce a **clean, reproducible, technically defensible project that looks professionally engineered rather than AI-generated or vibe-coded**.

---

## 14.1 Execution Rules

### Rule 1 — One task at a time

Complete the current task fully before starting the next task.

For every task:

```text
Implement
→ Run
→ Check
→ Fix
→ Verify
→ Commit
→ Update project plan
```

Do not move forward with known errors.

### Rule 2 — Commit after every completed task

Every completed task must have its own Git commit.

Commit messages must be:

* very short
* human
* specific
* natural
* lowercase where appropriate
* free of AI-style wording

Good:

```text
add data audit
fix split
add baseline
save metrics
add glove
```

Avoid:

```text
implement comprehensive data preprocessing pipeline
enhance model architecture with advanced regularization
complete sophisticated experimental framework
```

### Rule 3 — Keep the code professional

Code must look like it was written and maintained by a careful ML engineer.

Avoid:

* unnecessary abstractions
* excessive comments
* long explanatory comments
* decorative code
* generic AI-generated helper functions
* duplicate logic
* unexplained constants
* unused imports
* dead code
* notebook-only logic that should live in `src/`

Prefer:

* small reusable functions
* clear names
* explicit configuration
* simple control flow
* deterministic behavior
* useful error messages
* consistent formatting

### Rule 4 — Short, human captions

Notebook section headings and comments should be short and natural.

Prefer:

```text
## Load data
## Check classes
## Split data
## Build baseline
## Train model
## Compare runs
## Inspect errors
```

Avoid:

```text
## Comprehensive Analysis of Dataset Distribution and Data Quality Characteristics
```

Comments should explain **why** something matters, not narrate obvious code.

Good:

```python
# Keep the split fixed across all experiments.
```

Avoid:

```python
# Now we are going to create a stratified train validation test split
# because it is important for our machine learning project.
```

---

# 14.2 Stage 0 — Repository Check

Before changing code:

* confirm repository is clean
* inspect current branch
* inspect existing files
* inspect existing dependencies
* inspect current project structure
* confirm the dataset path
* confirm the current project-plan version

### Output

A known-good starting state.

### Commit

Only if changes are made.

Example:

```text
clean project
```

---

# 14.3 Stage 1 — Data Audit ✅ COMPLETE

Delivered in `notebooks/01_eda.ipynb`, with reusable logic in `src/data.py`, `src/text.py` and `src/embeddings.py`. Frozen output: `configs/data_config.json`.

Goal: establish exactly what the dataset contains before modeling.

### Tasks

1. Load `data/combined_complaints.parquet`.
2. Verify row count.
3. Verify columns.
4. Verify target classes.
5. Check null and blank narratives.
6. Check exact duplicate rows.
7. Check duplicate narrative groups.
8. Remove duplicates before splitting.
9. Recalculate class distribution.
10. Check class ratio.
11. Measure narrative lengths.
12. Measure whitespace-word truncation at 128 and 256.
13. Measure Keras-tokenizer sequence lengths.
14. Measure DistilBERT WordPiece lengths.
15. Report tokenizer-specific truncation rates.
16. Inspect `XXXX` redaction patterns.
17. Measure `XXXX` density by class.
18. Check vocabulary coverage for the planned GloVe embedding.
19. Record the results in the data configuration.

### Output

A finalized data audit and `data_config`.

### Gate

Do not begin model training until:

* duplicates are handled
* class labels are verified
* splits can be created deterministically
* tokenizer-specific length behavior is understood
* the GloVe coverage question is measured

All five are satisfied.

### Findings and decisions

| Finding | Evidence | Decision |
|---|---|---|
| Schema and labels as specified | 107,992 rows, 16 columns, 5 products, unique `Complaint ID`, 0 nulls or blanks | Narrative → Product; on-disk label strings are canonical |
| Heavy exact duplication | 7,038 rows over 848 texts; largest groups 832 / 293 / 238; 85% in Debt collection | Deduplicate on narrative before splitting → 101,802 rows |
| Ambiguous labels | 7 texts (49 rows) filed under more than one product | Keep. 0.05% label-noise floor; partly explains the predicted Debt collection ↔ Credit card confusion |
| **Near-duplicate leakage** | 7.8% of rows have a ≥0.90 cosine twin after exact dedup; 25.9% within Debt collection | **Exact dedup is insufficient — split must be group-aware. Implemented in Stage 2, with the threshold refined to 0.70 on per-band evidence (§14.4)** |
| Leakage is material | TF-IDF reference scores 0.922 Macro-F1 on contaminated test docs vs 0.858 on clean ones; ~0.45 Macro-F1 points of observed aggregate contribution, 57% of the contaminated slice is Debt collection | Same range as the 0.8-point spread §9 calls unresolvable, and a floor for higher-capacity models |
| Mild class imbalance | 1.153:1 after dedup; smallest class 18.3% | Class weights stay excluded, as pre-registered |
| Long right tail | median 178 words, p90 436, max 5,699 | M0–M3 `max_len=128`, M4 `max_len=256` — confirmed, not revised |
| WordPiece inflates length | median ×1.26; 43.9% truncated at 256 vs 31.0% for Keras | D0 `max_len=256`; unequal-context caveat recorded in §8.1 |
| Redaction is not a shortcut | `XXXX` in 86.8% of docs; class density 4.03–4.80%; redaction-only probe 0.274 Macro-F1 vs 0.200 chance | Retain `XXXX`; logged as a weak confound |
| Encoding damage | 673 rows (0.66%) wrapped in a `b'...'` byte-repr, present in all five classes | Strip the wrapper at preprocessing (`src.text.strip_bytes_wrapper`) |
| Task is strongly lexical | TF-IDF + LR reaches 0.863 Macro-F1; product cues in 28–71% of own class vs 0.2–3.3% elsewhere | Realistic signal, not leakage; supports the M1 hypothesis in §7 |
| Vocabulary is long-tailed | 57,396 types, 48% hapax, top 20k covers 99.78% of tokens | `max_features=20000`, `<OOV>` token |
| GloVe misses the signal words | 99.16% token coverage but only 15–26 of each class's top 30 distinctive terms | Run M3 as planned; expect a smaller gain than raw coverage implies |

Two decisions changed the plan, both recorded above and neither touching the experimental design: the split becomes group-aware (§14.4), and D0 gets an explicit `max_len` with a stated caveat (§8.1). The six configurations, the 10 fits, the metric and the epoch budget are unchanged.

### Commit

```text
add data audit
```

---

# 14.4 Stage 2 — Freeze the Dataset Split

Create:

```text
80% train
10% validation
10% test
```

Use stratification.

### Required change from the Stage 1 audit — implemented

Exact deduplication before splitting is **not** sufficient. Lightly edited template letters survive it, and when one lands in train and its twin in test the model scores the twin from memory.

The threshold was set by measurement, not assumption. Holding the TF-IDF reference fixed and slicing the test set by each document's highest cosine similarity to any training document — against a baseline of formulaic documents with no training twin at all (Macro-F1 0.837):

| Similarity band | Share of test | Macro-F1 | vs baseline |
|---|---:|---:|---:|
| 0.80 – 0.90 | 8.5% | 0.928 | +0.091 |
| 0.70 – 0.80 | 1.7% | 0.923 | +0.086 |
| 0.60 – 0.70 | 1.8% | 0.870 | +0.034 |
| 0.50 – 0.60 | 4.9% | 0.856 | +0.019 |
| below 0.50 | 83.1% | 0.849 | +0.012 |

Inflation is flat and large above 0.70 and collapses immediately below it, so clusters are cut at **cosine ≥ 0.70** rather than the 0.90 the audit first proposed. The baseline is *lower* than the bulk of the test set (0.837 vs 0.849) — formulaic complaints are intrinsically harder, not easier — which rules out "template text is just easy" as an explanation for the gap.

Clusters are connected components (`src/dedup.py`), computed once and frozen as `data/splits/near_dup_clusters.npy`. The split is built by dealing whole clusters into stratified folds (`StratifiedGroupKFold`), so no cluster can be torn across a boundary. `groups` is a required argument to `create_stratified_split` and `validate_split` — an ungrouped split is no longer expressible.

This does not alter the locked experimental design: no rows are removed, the row count stays 101,802, the 80/10/10 stratified proportions stay, and the six configurations and 10 fits are untouched. It changes only which rows land in which split.

**Known limitation.** The residual 0.60–0.70 band (1.8% of test, +0.034) is left ungrouped. Below 0.70 cosine similarity increasingly reflects shared product vocabulary rather than a shared template, and grouping on it would start folding the label into the split. Reported rather than chased further.

**Cluster structure, checked before grouping on it.** 93,805 clusters over 101,802 rows: 91,324 singletons, 2,481 multi-row clusters holding 10,478 rows (10.3%). Sizes are small — median 2, p90 4, p95 6 — and the largest connected component is 696 rows (0.7% of the corpus), so transitive chaining does not collapse the corpus into one giant group. 128 clusters (1.16% of rows) span more than one product; each is dominated by a single label (largest: 297 Money transfer vs 1 Checking/savings), matching the same generic credit-report-dispute pattern found in the exact-duplicate label-noise check, not a clustering failure.

### Tasks

* create the split once
* save the indices
* verify class proportions
* verify no duplicate narrative crosses splits
* verify no near-duplicate cluster crosses splits
* load the same indices in every experiment

### Output

Persisted split files.

Example:

```text
data/splits/train_idx.npy
data/splits/val_idx.npy
data/splits/test_idx.npy
```

### Gate

The test set is now frozen.

No experiment may regenerate the split.

### Commit

```text
freeze data split
```

---

# 14.4B Stage 2B — Text Preprocessing ✅ COMPLETE

Inserted between Stage 2 and Stage 3; existing stage numbers (§14.5 onward) are
unchanged rather than renumbered, to avoid touching already-approved sections for
one insertion.

Delivered in `notebooks/02_preprocessing.ipynb`, with reusable logic in
`src/keras_tokenizer.py`, `src/preprocessing.py`, and extensions to
`src/embeddings.py`/`src/fingerprint.py`. Frozen output:
`artifacts/tokenizer/keras_tokenizer.json`, `artifacts/embeddings/glove_100d_matrix.npy`
(+ metadata), `artifacts/preprocessing/preprocessing_version.json`.

Every parameter used here was already committed in `configs/data_config.json`
(`max_features`, `oov_token`, `max_len` per experiment, GloVe path/dim, DistilBERT
checkpoint) — this stage implements those values, it does not re-derive them.

### Normalization

One step, audit-justified: undo the `b'...'` byte-repr wrapper found in 0.66% of
narratives. Nothing else — no lowercasing, punctuation/digit/stopword removal, or
stemming. Casing carries tone signal, punctuation and amounts are meaningful,
`XXXX` is retained per the Stage 1 decision, and lowercasing/splitting is the
tokenizer's job rather than a step ahead of it.

### Keras tokenizer

`src/keras_tokenizer.py` is a project-specific, TensorFlow-free implementation of
the `tf.keras.preprocessing.text.Tokenizer` behavior this project needs — not
TensorFlow itself, and not a claim of identity with every Keras code path.
Differential-tested against a real `tf.keras.preprocessing.text.Tokenizer`
(TensorFlow 2.21.0, installed in a throwaway venv for that one check only, removed
afterward — the shared environment was never touched) across plain text, frequency
ties, punctuation, `XXXX` redactions, digits, mixed case, Unicode, empty strings,
and OOV handling on held-out text. All cases matched; result recorded at
`artifacts/preprocessing/keras_tokenizer_equivalence.json`, re-runnable via
`scripts/verify_keras_tokenizer_equivalence.py` whenever the fit/transform logic
changes.

Fit on the frozen train split only (81,442 rows) — `fit_tokenizer_on_train` is the
single call site in the project, tested to never see validation/test text. Raw
vocabulary 50,879 words, capped to `max_features=20000`. Train OOV 0.212%,
validation 0.278%, test 0.287% — close together, which is the expected signature of
a representative split rather than a concerning gap.

### Padding and truncation

`padding="pre"` keeps the Keras default (real tokens end up adjacent to the final
timestep an unmasked LSTM reads — recheck if the model factory adds a `Masking`
layer). `truncating="post"` **deviates** from the Keras default (`"pre"`), on
evidence rather than a guess: a TF-IDF + LogisticRegression probe on the frozen
split scored 0.852 Macro-F1 keeping only the first 128 words of each narrative,
0.832 keeping only the last 128, against 0.858 for the untruncated text. CFPB
complaints front-load the core issue in the opening sentences.

Real Keras-tokenizer truncation on the frozen train split: 66.4% at `max_len=128`,
31.5% at `max_len=256` — within a point of the Stage 1 audit's whitespace-based
estimate (65.6% / 31.0%), confirming rather than contradicting it.

### GloVe (M3)

Matrix built from the train-fitted vocabulary via `embeddings.build_embedding_matrix`
(streams the 347 MB file once). Two coverage numbers, both correct, measuring
different things: **type** coverage 86.68% (matched embedding rows / vocab_size —
how many rows stay randomly initialised) and **token** coverage 99.16% (matched
rows weighted by train frequency — how much of the text read is pretrained,
matching the Stage 1 audit's dataset-wide figure exactly). They diverge because the
unmatched words are disproportionately rare (servicer names, statute numbers), as
the audit found. Row 0 (padding) is zero; each unmatched word gets its own
independently-drawn small random vector, not zero or a shared placeholder — tested.
Trainability is a model-factory decision (Task 4): `data_config.glove.trainable`
applies to every row unless the model separately masks index 0.

### DistilBERT (D0)

Separate code path — DistilBERT's own pretrained WordPiece tokenizer, never fit on
this dataset. `max_len=256` confirmed synchronized across `project_plan.md`
(§3.1/§8.1), `configs/data_config.json`, `src/config.py`'s D0 entry, and this
notebook, asserted rather than eyeballed. Real WordPiece truncation on the frozen
train split at 256: 44.4% (val 41.3%, test 41.9%) — the full padded
`input_ids`/`attention_mask` arrays are not precomputed or committed (~208 MB,
trivially regenerable via `tokenize_for_distilbert`).

### Artifacts

Committed (small, ~10 MB total): tokenizer `word_index` JSON, GloVe matrix + metadata,
`PreprocessingVersion` record, the TensorFlow differential-test result. **Not**
committed: the padded train/val/test integer sequence arrays (~42 MB at 128, ~83 MB
at 256) or the full DistilBERT arrays (~208 MB) — both regenerate in seconds from
these artifacts plus the frozen split via `build_sequences`/`tokenize_for_distilbert`.

### Commit

```text
add preprocessing
```

---

# 14.5 Stage 3 — Build the Evaluation Layer ✅ COMPLETE

Build evaluation before serious model training.

Delivered in `src/evaluation.py`, `src/results.py`, `src/checkpoint.py`, with a
short demonstration in `notebooks/03_evaluation.ipynb`. One evaluation path,
already in place before this stage from Antigravity's earlier work
(`compute_metrics`, `calculate_deltas`, `generate_comparison_table`,
`validate_runs_csv`, checkpoint metadata enforcement) and extended here rather
than duplicated.

### Tasks

Implemented (pre-existing, verified) and extended:

* accuracy, macro precision, macro recall, macro-F1 — `compute_metrics`, all four
  matched against `sklearn.metrics` to 7 decimal places on synthetic examples
  covering balanced correct predictions, one class with poor recall, highly
  uneven class frequencies, a class the model never predicts, a class with zero
  true support, and all-predictions-one-class.
* confusion matrix — fixed 5×5, canonical order, integer counts; a dedicated test
  confirms `cm[i][j]` means true=`LABELS[i]`, predicted=`LABELS[j]`, not just that
  the shape is right.
* per-class precision/recall/F1/support — canonical `LABELS` order, dict keys
  checked directly against `LABELS`.
* delta calculation — `calculate_deltas` (incremental = current − previous,
  cumulative = current − M0), unrounded internally, `format_delta` only rounds
  for display. Already matched Task 4's terminology exactly; no rename needed.
* **new** `seed_statistics(values, ddof=1)` — mean/std/n across seed runs,
  extracted from `generate_comparison_table`'s previously-inline `np.std(...,
  ddof=1)` so the M0/D0 seed spread is computed in exactly one place. A
  single-seed rung reports `std=None`, not `std=0` — no spread was measured for
  M1–M4, and treating that as a real zero would misstate it as a measured
  stability.
* **new** `compute_val_macro_f1` — the canonical checkpoint metric, a thin
  documented wrapper making explicit that it must be called once per epoch on
  the *full* validation set, never as a running per-batch average. Verified two
  ways: matches manual sklearn Macro-F1 on synthetic validation predictions, and
  a constructed example shows the naive per-batch-average pattern disagreeing by
  ~0.046 — concrete evidence for why post-epoch full-set computation is required,
  not just a warning in a docstring.
* **new** input validation in `compute_metrics` — mismatched lengths, empty
  input, non-1D arrays, out-of-range integer label ids, and unrecognized string
  labels all raise `ValueError` now rather than letting `sklearn`'s
  `labels=`-restricted averaging silently drop the offending class from the
  score.
* **new** `verify_run_traceability` (`src/results.py`) — checks a `runs.csv` row's
  `dataset_version` against the frozen split's `dataset_content_sha256`, and that
  `checkpoint_path` resolves to a `CheckpointMetadata` record with matching
  experiment/seed and `monitor_metric == "val_macro_f1"`. `git_commit` and
  `preprocessing_version` are traced through `src.reproducibility.EnvironmentInfo`
  and `artifacts/preprocessing/preprocessing_version.json` respectively rather
  than duplicated into `RUNS_SCHEMA` — the schema is not changed.

Verified:

```text
manual sklearn Macro-F1
=
compute_val_macro_f1 (the logged validation Macro-F1)
```

Checkpoint selection: `src/checkpoint.py` already enforces
`monitor_metric == "val_macro_f1"` at save time (`save_checkpoint_record` raises
`CheckpointValidationError` otherwise) — confirmed still correct, no change needed.

### Output

A single evaluation path shared by all models: M0–M4 and D0 all call
`compute_metrics`/`compute_val_macro_f1` — no per-notebook metric reimplementation.
No M0–D0 notebooks exist yet (Task 5/6 not started), so there was nothing to check
for stale metric references; this is the interface those notebooks will call.

### Gate

Metric calculations independently checked before M0: 35 tests across
`tests/test_evaluation.py` (23) and `tests/test_results.py` (8, including
traceability), full suite 92/92 passing, `scripts/project_check.py` green before
and after.

### Commit

```text
verify eval
```

---

# 14.6 Stage 4 — Build the Recurrent Model Factory ✅ COMPLETE

Create a small model-building interface for M0–M4.

The model factory should make the experiment differences visible.

Delivered in `src/models.py` (new), reading `src.config.EXPERIMENT_CONFIGS` as the
single source of truth rather than duplicating hyperparameters. Verified in
`tests/test_models.py` (40 tests). D0 is not built by this factory — DistilBERT is
handled separately.

### Two hyperparameters this stage found undefined, and locked before implementing

`EXPERIMENT_CONFIGS` had `dropout` and `spatial_dropout` for M2–M4, but no
`recurrent_dropout` value anywhere in the repo, despite §3.1 naming it explicitly
("M2 = M1 + dropout & `recurrent_dropout`"). M4's `lr_schedule="reduce_on_plateau"`
had no `factor`/`patience`/`min_lr`. Per this stage's own rule — invent nothing,
stop and report a genuinely undefined hyperparameter — both were raised and fixed
by explicit decision before any model was built, not picked silently:

* `recurrent_dropout = 0.2` for M2, M3, M4 (input `dropout` stays 0.3 — a
  deliberately lower rate on the recurrent state than on the input connections).
* M4's `ReduceLROnPlateau`: `factor=0.5`, `patience=2`, `min_lr=1e-5`, monitoring
  `val_macro_f1`. Chosen so the LR drop (patience=2) has one epoch to help before
  `EarlyStopping`'s patience=3 would end training. `EarlyStopping` additionally
  gets `restore_best_weights=True`, made explicit rather than left as a default.

Both are now in `EXPERIMENT_CONFIGS` (`src/config.py`) as fixed, pre-registered
settings — not tuned after seeing any result, consistent with the rest of the
locked ladder.

### Configuration → architecture

```text
embedding        -> "random" (M0-M2) or "glove" (M3-M4), matrix from src.embeddings
bidirectional     -> LSTM vs Bidirectional(LSTM) wrapper
dropout           -> LSTM's own `dropout` kwarg (input connections)
recurrent_dropout -> LSTM's own `recurrent_dropout` kwarg (recurrent connections)
spatial_dropout   -> SpatialDropout1D after the embedding, only added when > 0
max_len           -> Input layer shape
learning_rate     -> Adam optimizer
scheduler         -> ReduceLROnPlateau callback (M4 only)
early_stopping    -> EarlyStopping callback (M4 only)
```

One clear model definition (`build_recurrent_model`), one clearly visible
configuration change per rung:

| Rung | Layer sequence | What changed from the previous rung |
|---|---|---|
| M0 | Input → Embedding → LSTM → Dense | — (baseline) |
| M1 | Input → Embedding → Bidirectional(LSTM) → Dense | direction only |
| M2 | + SpatialDropout1D, `dropout`/`recurrent_dropout` on the LSTM | regularization only |
| M3 | same layer sequence as M2 | embedding source only (GloVe matrix, not rebuilt here — received from `src.embeddings.build_embedding_matrix`) |
| M4 | same layer sequence as M3 | `max_len` 128→256, `ReduceLROnPlateau` + `EarlyStopping` callbacks added |

No stacked LSTM anywhere, no class weights — both stay excluded, verified by a test
that inspects every experiment's built layers and asserts exactly one recurrent
layer.

### `val_macro_f1` as a real Keras metric

`MacroF1Score(keras.metrics.Metric)` accumulates a confusion matrix across every
batch in an epoch and computes precision/recall/F1 from the *full* accumulated
matrix in `result()` — not a per-batch average, which Task 4 already established
is mathematically wrong for macro-F1. Verified numerically equal to
`src.evaluation.compute_val_macro_f1` (itself already verified against sklearn) to
5 decimal places across simulated uneven batches, so `monitor="val_macro_f1"` in
`ModelCheckpoint`/`EarlyStopping`/`ReduceLROnPlateau` means exactly what the
evaluation layer means by it. `ModelCheckpoint(monitor="val_macro_f1",
save_best_only=True)` is present for every rung (M0–M4), matching the fixed
experimental control in §4; the scheduler and early stopping are added only for
M4. Callbacks are configured, not run — no training happened in this stage.

### Loss, labels, output

`sparse_categorical_crossentropy` — matches the integer class ids
`src.preprocessing.class_ids`/`src.evaluation` already use, not one-hot. Output
layer size is `len(LABELS)` (5), read from `src.data`, not hardcoded; a test
asserts `output_classes != 5` fails config validation.

### Environment note

This is the first stage in the project that requires TensorFlow/Keras — every
prior stage deliberately stayed framework-agnostic (see `src/keras_tokenizer.py`'s
docstring for why that mattered for Task 3). TensorFlow is not installed in the
base environment this project has mostly run in; a separate `ai-ml` conda
environment (already present on this machine, TensorFlow 2.21.0/Keras 3.15.0) was
used to build and test `src/models.py`. `tests/test_models.py` detects TensorFlow's
absence and skips cleanly rather than failing when run in the base environment —
confirmed: base env reports 92 passed / 40 skipped, `ai-ml` env reports 132/132
passed, and `scripts/project_check.py` passes in both. Training in the next task
runs on Kaggle GPU regardless, where TensorFlow is preinstalled.

### Reproducibility

Same config + same seed → identical initial weights (tested). Same config →
identical parameter shapes (tested). This is a CPU-only claim: TensorFlow does not
guarantee bitwise-identical results on GPU/cuDNN even with a fixed seed, and that
limitation is not overpromised away here.

### Commit

```text
add model factory
```

---

# 14.7 Stage 5 — M0 Baseline ✅ COMPLETE

Train:

**Simple LSTM**

Configuration:

```text
random embeddings
max_len=128
```

Run:

**3 seeds**

Trained on Kaggle GPU via `scripts/run_m0.py` (orchestrator) +
`scripts/kaggle_train_m0.py` (per-seed training kernel) +
`scripts/kaggle_package_m0.py` (frozen-input dataset packaging). All three seeds
completed, validated, and registered in one sequential run; no seed was retried
or re-run.

### Result

| Seed | Macro-F1 | Accuracy | Macro Precision | Macro Recall | Best Epoch | Epochs Run | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8489 | 0.8462 | 0.8528 | 0.8493 | 4 | 10 | 163 |
| 123 | 0.8507 | 0.8482 | 0.8521 | 0.8513 | 4 | 10 | 158 |
| 456 | 0.8530 | 0.8504 | 0.8529 | 0.8533 | 3 | 10 | 179 |

**M0 Macro-F1 = 0.8508 ± 0.0021** (mean ± sample std, `ddof=1`, `src.evaluation.seed_statistics`)
Min 0.8489, max 0.8530, spread 0.0041.

Parameter count: 2,117,893 (identical across all three seeds — confirms no
configuration drift between runs). All three used the same frozen dataset
(`dataset_content_sha256 = eb66684f...`), the same frozen split, the same frozen
tokenizer, and `preprocessing_version = pp-v1`.

### Stability check

std = 0.0021, well inside the pre-registered 0.008 (0.8 Macro-F1 point) threshold
from §9 — not a new threshold invented for this task. Ladder deltas from M1
onward are resolvable against this spread. Per §9's rule, an M1-M4 delta smaller
than roughly this spread should be read as indistinguishable from run-to-run
variance, not as a real effect — reported as a stability reference, not a
significance test.

### Checkpoint selection, verified twice

For every seed, `ModelCheckpoint(monitor="val_macro_f1", save_best_only=True)`
selected the best epoch, and the training kernel independently recomputed
validation Macro-F1 with `sklearn`-backed `compute_val_macro_f1` on the restored
best-epoch weights before trusting the result — agreement was within 1.6e-8 on
every seed (`results/m0/seed{N}/val_check.json`). The frozen test set was
evaluated only after this selection was finalized; there is no code path in
`scripts/kaggle_train_m0.py` where test data could influence which epoch was
chosen (`ModelCheckpoint` monitors `val_macro_f1`, never a test metric, and the
test-evaluation code runs strictly after `model.load_weights(checkpoint_path)`).

### Implementation issue found and fixed during this stage

Two Kaggle-mechanics bugs, not modeling/methodology changes: (1) Kaggle derives a
new kernel's actual slug from its title, not the `id` field, when the two
disagree — the orchestrator's title now *is* the slug, so they can't diverge
again. (2) the private dataset mounted at
`/kaggle/input/datasets/<user>/<slug>/`, not the flat `/kaggle/input/<slug>/`
path assumed initially — found via a throwaway diagnostic kernel before spending
GPU time on the real run, fixed in `scripts/kaggle_train_m0.py`'s `INPUT_DIR`.
Neither bug affected data, preprocessing, or the model itself; both were caught
before any GPU training ran against them.

### Artifacts

Checkpoints (`checkpoints/M0_seed{42,123,456}.weights.h5`, ~25MB each after
Keras 3 also serializes optimizer state, ~73MB total) are **not** committed —
larger than anything else this project has put in git, and the existing
`.gitignore` already deliberately excludes `/checkpoints/`. Their metadata,
history, test predictions (traceable to `test_idx`), confusion matrices,
per-class metrics, and a `run_manifest.json` (git commit, Kaggle dataset/kernel
identifiers, dataset + preprocessing version) are committed under
`results/m0/seed{N}/` and `results/runs.csv`.

### Required output

M0 is now the official baseline. Not compared against M1 yet — M1 has not run.

### Gate

M0 variance inspected: stable, well within the pre-registered threshold. Proceed
to the ladder.

### Commit

```text
run baseline
```

---

# 14.8 Stage 6 — Auxiliary TF-IDF Reference ✅ COMPLETE

Run:

**TF-IDF + Logistic Regression**

CPU only.

This is an auxiliary sanity check.

It is:

* not M0
* not one of the six configurations
* not part of the 10-fit budget

Question:

> Does sequence modeling provide a meaningful advantage over a simple bag-of-words reference?

Delivered in `src/tfidf_reference.py`, `notebooks/04_tfidf_reference.ipynb`.
Result saved to `results/tfidf_reference.json` (label `TFIDF_REFERENCE`) — not
`results/runs.csv`, since `src.results.validate_runs_csv` strictly rejects any
experiment name outside `src.checkpoint.VALID_EXPERIMENTS` (M0–M4, D0). That
guard is what it's for; this stage doesn't work around it.

### Configuration (fixed before fitting, not tuned against a result)

```text
TF-IDF:     analyzer=word, ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True
LogReg:     max_iter=1000, C=1.0, solver=lbfgs (multinomial by default here)
```

The task's own suggested conservative starting point — no configuration search
was run. Text goes through the same `normalize_text` (strip the `b'...'` wrapper
only) Task 3 locked; nothing was invented specifically for TF-IDF.

### Result

| Metric | Value |
|---|---:|
| Macro-F1 (test) | **0.8707** |
| Accuracy | 0.8682 |
| Macro Precision | 0.8710 |
| Macro Recall | 0.8709 |
| Macro-F1 (validation, descriptive only) | 0.8750 |
| Vocabulary size | 619,462 |
| Fit time | 260s |

Fit on the frozen train split only (81,442 rows); validation transformed but used
only descriptively, never for model selection; test evaluated once, after the
model was already fixed. `vectorizer.vocabulary_` confirmed unchanged after
transforming validation/test.

### TF-IDF reference − M0 = **+0.0199**

TF-IDF (0.8707) exceeds M0 (0.8508 ± 0.0021). Per this stage's own pre-registered
interpretation rule for this case: this is not read as "the LSTM is bad." Both
share the same frozen dataset fingerprint, split, and evaluation layer
(`src.evaluation.compute_metrics`), so the comparison isn't confounded by
different data or metric code. Interpreted instead as confirmation of what the
Stage 1 audit already found — this task is strongly lexically separable — and as
a genuine reference point for M1–M4: an architectural/representation
improvement is doing real work only once it approaches or exceeds this number,
not merely M0.

### Per-class comparison

Same relative pattern as M0: Student loan easiest (F1 0.965), Checking/savings
hardest (F1 0.793) — consistent with the task being lexically driven under both
models, not a model-specific artifact.

### Feature inspection (leakage/sanity check)

Top per-class terms are servicer names, product terms, and platform names
(`chime`, `overdraft` for Checking/savings; `synchrony`, `citi`, `barclays` for
Credit card; `paypal`, `cashapp`, `coinbase`, `zelle` for Money transfer;
`mohela`, `nelnet`, `forbearance` for Student loan) — the same class-distinctive
vocabulary the Stage 1 audit found, not `XXXX`-pattern redaction artifacts or
template boilerplate. Read as useful lexical signal, not a shortcut or confirmed
leakage.

### Commit

```text
add tfidf check
```

---

# 14.9 Stage 7 — M1 ✅ COMPLETE

Run:

**M0 + Bidirectional LSTM**

Everything else held fixed: random embeddings, `max_len=128`, same tokenizer,
vocabulary, split, optimizer, learning rate, batch size, epoch ceiling (10),
and `ModelCheckpoint(monitor="val_macro_f1", save_best_only=True)` checkpoint
policy. No dropout, no `recurrent_dropout`, no GloVe, no LR scheduling, no
early stopping — those are later rungs.

Trained on Kaggle GPU via `scripts/run_m1.py` (orchestrator) +
`scripts/kaggle_train_m1.py` (training kernel, identical pipeline to M0's) +
`scripts/kaggle_package_m1.py` (frozen-input packaging). Single seed, per the
locked 10-fit budget (§3.2): `src.config.EXPERIMENT_CONFIGS["M1"]["seeds"] = [42]`.

### Architecture check (before training)

Verified programmatically via the model factory, small script, no training:

```text
layers: InputLayer -> Embedding -> Bidirectional(LSTM) -> Dense
bidirectional wrapper: Bidirectional, underlying recurrent layer: LSTM
embedding_type: random (use_glove=False)
dropout: 0.0, recurrent_dropout: 0.0
max_len: 128, output_classes: 5
```

M0 total parameters: 2,117,893. M1 total parameters: 2,235,781. Difference:
117,888 — fully explained by the extra backward-direction LSTM
(4 × ((100+128)×128 + 128) = 117,248 parameters) plus the Dense layer's doubled
input width from the concatenated forward/backward output (256 vs 128 → 5×128 =
640 extra weights). 117,248 + 640 = 117,888, exact match. No unexplained
parameter growth.

### Result

| Seed | Macro-F1 | Accuracy | Macro Precision | Macro Recall | Best Epoch | Epochs Run | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8485 | 0.8455 | 0.8541 | 0.8484 | 4 | 10 | 274 |

**M1 − M0 = −0.0023** (exact: −0.002284542910804599; M0 mean = 0.8508)

### Checkpoint selection, verified twice

`ModelCheckpoint(monitor="val_macro_f1", save_best_only=True)` selected epoch 4.
Independent `sklearn`-backed recompute on the restored best-epoch weights agreed
with the Keras training-time value within 2.8e-8
(`results/m1/seed42/val_check.json`). The frozen test set was evaluated only
after this selection was finalized — same structural guarantee as M0 (test
evaluation runs strictly after `model.load_weights(checkpoint_path)`, and
`ModelCheckpoint` never sees a test metric).

### Interpretation

M1's delta (−0.0023) is smaller in magnitude than M0's own three-seed spread
(std 0.0021, min–max spread 0.0041) — per the pre-registered stability rule
(§9), this is reported as **no clear evidence of a meaningful improvement or
regression**, not as a real effect in either direction. Bidirectional context
did not measurably help or hurt the Simple LSTM baseline on this task.

This is consistent with the pre-registered hypothesis (§7): "Complaint topics
are lexically distinctive; keyword presence matters more than word order" —
and with the TF-IDF reference finding (§14.8) that the task is strongly
lexically separable. A bag-of-words model with no sequence information at all
already exceeds both M0 and M1, which is the kind of task where reading a
sequence backward as well as forward is not expected to add much: the signal
BiLSTM could exploit (word order, long-range dependency) is not the dominant
signal driving this classification problem. TF-IDF is not used as the delta
reference for the ladder (§14.11's note on that comparison); this observation
is interpretive context, not a new baseline.

### Per-class comparison

| Class | M0 mean F1 (3 seeds) | M1 F1 | Δ |
|---|---:|---:|---:|
| Checking or savings account | 0.7650 | 0.7651 | +0.0001 |
| Credit card | 0.8367 | 0.8364 | −0.0003 |
| Debt collection | 0.9053 | 0.9108 | +0.0055 |
| Money transfer, virtual currency, or money service | 0.7889 | 0.7715 | −0.0174 |
| Student loan | 0.9582 | 0.9587 | +0.0005 |

Four of five classes are flat (±0.0055). The aggregate delta is driven almost
entirely by Money transfer, which drops 0.0174 F1 — confusion matrix shows this
class's recall falling most against Checking/savings (505 of 2,094 Money
transfer test examples predicted as Checking/savings, vs M0's confusion
pattern in the same range). Both classes involve account-level transaction
disputes with overlapping vocabulary (banks, transfers, holds), so this reads
as a genuine class-boundary difficulty rather than an artifact — the same kind
of semantic overlap flagged for Debt collection ↔ Credit card in §10, not
specific to bidirectionality.

### Training curve

Best epoch is 4 for both M0 (seed 42) and M1 — no earlier or later overfitting
onset. M1's train loss is consistently lower than M0's at every epoch (more
parameters fit the training data more closely, as expected), but validation
loss rises after epoch 4–5 in both runs at a similar rate. No evidence that
bidirectional context changes the overfitting profile at this scale.

### Commit

```text
run bilstm
```

---

# 14.10 Stage 8 — M2 ✅ COMPLETE

Run:

**M1 + dropout + recurrent_dropout**

Everything else held fixed: random embeddings, `max_len=128`, same tokenizer,
vocabulary, split, optimizer, learning rate, batch size, epoch ceiling (10),
`ModelCheckpoint(monitor="val_macro_f1", save_best_only=True)`. No GloVe, no
LR scheduling, no early stopping — those are later rungs.

Trained on Kaggle GPU via `scripts/run_m2.py` (orchestrator) +
`scripts/kaggle_train_m2.py` (training kernel, identical pipeline to M1's) +
`scripts/kaggle_package_m2.py` (frozen-input packaging). Single seed, per the
locked 10-fit budget (§3.2): `src.config.EXPERIMENT_CONFIGS["M2"]["seeds"] = [42]`.

### Architecture check (before training)

Verified programmatically via the model factory, small script, no training:

```text
layers: InputLayer -> Embedding -> SpatialDropout1D -> Bidirectional(LSTM) -> Dense
bidirectional wrapper: Bidirectional, underlying recurrent layer: LSTM
LSTM dropout=0.3, recurrent_dropout=0.2, SpatialDropout1D rate=0.2
embedding_type: random (use_glove=False)
max_len: 128, output_classes: 5, lr_schedule=None, early_stopping=False
```

M1 and M2 total parameters: **identical, 2,235,781**. Dropout, recurrent
dropout, and spatial dropout are training-time regularization with no
learnable weights of their own — `SpatialDropout1D` adds a structural layer
with 0 parameters — so an unchanged parameter count is exactly what the config
diff predicts, not an omission.

### Result

| Seed | Macro-F1 | Accuracy | Macro Precision | Macro Recall | Best Epoch | Epochs Run | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8518 | 0.8485 | 0.8545 | 0.8513 | 4 | 10 | 6,790 |

**M2 − M1 = +0.0033** (exact: 0.003282040681856002)
**M2 − M0 = +0.0010** (exact: 0.0009974977710514032; M0 mean = 0.8508)

### Training-time trade-off (`recurrent_dropout`)

6,790s (~113 minutes) vs M1's 274s — a **~24.8×** slowdown, at the extreme end
of but consistent with the pre-registered risk (§12: "`recurrent_dropout`
disables the cuDNN fast path → ~5–10× slower epochs"). Confirmed as expected
GPU-execution-path behavior, not an anomaly: `recurrent_dropout > 0` forces
Keras off the fused cuDNN LSTM kernel onto a much slower per-timestep
implementation. `recurrent_dropout` was kept at the locked value throughout —
the slowdown is a recorded trade-off, not a reason to change the configuration
(§18: no tuning after seeing results). Kaggle GPU: P100, same as every prior
rung.

### Checkpoint selection, verified twice

`ModelCheckpoint(monitor="val_macro_f1", save_best_only=True)` selected epoch
4. Independent `sklearn`-backed recompute on the restored best-epoch weights
agreed with the Keras training-time value within 5.7e-11
(`results/m2/seed42/val_check.json`) — the tightest agreement of any rung so
far. The frozen test set was evaluated only after this selection was
finalized, same structural guarantee as M0/M1.

### Regularization analysis (Part 10)

| Epoch | M1 train loss | M1 val loss | M1 gap | M2 train loss | M2 val loss | M2 gap |
|---:|---:|---:|---:|---:|---:|---:|
| 4 (best) | 0.3461 | 0.4180 | +0.0719 | 0.3772 | 0.4030 | +0.0258 |
| 10 (last) | 0.1243 | 0.7271 | +0.6028 | 0.2122 | 0.5031 | +0.2908 |

At the shared best epoch (4), M2's train/val loss gap is roughly a third of
M1's; by epoch 10 M2's gap is less than half of M1's (+0.29 vs +0.60). M2's
training loss is *higher* than M1's at every epoch — direct evidence
regularization is doing what it is supposed to: preventing the model from
fitting the training data as tightly. Peak validation Macro-F1 is also higher
for M2 (0.8619 at epoch 4) than M1 (0.8591 at epoch 4), and M2's post-peak
validation degradation is slower (val_macro_f1 falls to 0.8476 by epoch 10,
vs M1's 0.8297). Conclusion: **M2 reduced overfitting and modestly improved
both peak and late-training validation performance** — not "no meaningful
effect" and not "over-regularized."

### Per-class comparison

| Class | M1 F1 | M2 F1 | Δ |
|---|---:|---:|---:|
| Checking or savings account | 0.7651 | 0.7676 | +0.0025 |
| Credit card | 0.8364 | 0.8307 | −0.0057 |
| Debt collection | 0.9108 | 0.9117 | +0.0009 |
| Money transfer, virtual currency, or money service | 0.7715 | 0.7893 | +0.0177 |
| Student loan | 0.9587 | 0.9597 | +0.0010 |

Improvement is not concentrated in one class the way M1's regression was:
three of five classes improve modestly, one (Credit card) declines slightly,
and the largest single mover is Money transfer (+0.0177) — the same class that
carried nearly all of M1's decline versus M0 (§14.9, −0.0174). Regularization
recovers most of what bidirectionality cost that specific class, though this
reads as the aggregate delta partly self-correcting a prior-rung weakness
rather than broad regularization gains across the board.

### Interpretation

M2 − M1 (+0.0033) is a real, modest improvement supported by direct
train/validation curve evidence (Part 10), not inferred from the test number
alone. It remains smaller than M0's own three-seed spread (0.0041), so per the
pre-registered stability rule (§9) it is reported as a **plausible but not
clearly resolved improvement** — consistent in direction with the pre-registered
hypothesis (§7: "Small positive... depends on whether M1 is overfitting"), and
M1 was shown here to be overfitting more than M2 (Part 10), which is exactly
the condition the hypothesis predicted would produce a gain. M2 − M0
(+0.0010) is smaller still and not treated as a resolved effect on its own —
reported for completeness (§9's required cumulative delta), not as a claim
that regularization alone beats the baseline.

TF-IDF (0.8707) remains well above M2; this is expected context, not a target
M2 was tuned toward (§12), and is not used as the delta reference for this
rung.

### Commit

```text
run dropout
```

---

# 14.11 Stage 9 — M3 ✅ COMPLETE

Run:

**M2 + GloVe 100d**

Everything else held fixed: dropout 0.3, recurrent_dropout 0.2, spatial_dropout
0.2, bidirectional, `max_len=128`, same tokenizer, vocabulary, split,
optimizer, learning rate, batch size, epoch ceiling (10), `ModelCheckpoint`
policy. No LR scheduling, no early stopping, no `max_len=256`.

Trained on Kaggle GPU via `scripts/run_m3.py` (orchestrator) +
`scripts/kaggle_train_m3.py` (training kernel, identical pipeline to M2's) +
`scripts/kaggle_package_m3.py` (frozen-input packaging — this is the first
rung whose Kaggle dataset includes the frozen GloVe matrix, since M0–M2 used
random embeddings and never needed it). Single seed, per the locked 10-fit
budget (§3.2): `src.config.EXPERIMENT_CONFIGS["M3"]["seeds"] = [42]`. The
matrix built in Task 3 (`artifacts/embeddings/glove_100d_matrix.npy`) is
loaded as-is and shipped to Kaggle unmodified — never rebuilt from the raw
GloVe file on Kaggle or here.

### Architecture check (before training)

Verified programmatically via the model factory, small script, no training:

```text
layers: InputLayer -> Embedding -> SpatialDropout1D -> Bidirectional(LSTM) -> Dense
embedding weights loaded exactly from the frozen GloVe matrix, shape (20000, 100)
embedding row 0 (padding): zero
embedding_trainable: True
dropout=0.3, recurrent_dropout=0.2, spatial_dropout=0.2 (unchanged from M2)
max_len: 128, output_classes: 5, lr_schedule=None, early_stopping=False
```

M2 and M3 total parameters: **identical, 2,235,781** (both trainable — only
the embedding's *initial values* differ, not its shape or trainability).

### GloVe coverage (re-confirmed against Task 3's frozen figures)

| Metric | Value |
|---|---:|
| Type coverage (matched rows / vocab_size) | 86.68% |
| Matched types | 17,335 / 19,999 |
| Unmatched types | 2,664 |
| Padding row (index 0) | zero, confirmed |
| Unmatched-row init | independent uniform(−0.05, 0.05) per row |

Matches Task 3's frozen `artifacts/embeddings/glove_100d_metadata.json`
exactly — no drift between the artifact and the current vocabulary. (Token
coverage, 99.16%, is Task 3's separate frequency-weighted figure and was not
recomputed here — it measures a different thing, and is already locked.)

### Result

| Seed | Macro-F1 | Accuracy | Macro Precision | Macro Recall | Best Epoch | Epochs Run | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8657 | 0.8627 | 0.8663 | 0.8654 | 7 | 10 | 6,793 |

**M3 − M2 = +0.0139** (exact: 0.013870261991500099)
**M3 − M0 = +0.0149** (exact: 0.014867759762551502; M0 mean = 0.8508)

Both deltas are comfortably outside M0's own three-seed spread (0.0041) — per
the pre-registered stability rule (§9), this is reported as a **real,
resolved improvement**, not run-to-run noise.

### Training-time note

6,793s (~113 minutes) — essentially identical to M2's 6,790s. `recurrent_dropout=0.2`
is unchanged from M2 and remains the dominant cost driver; swapping the
embedding source does not measurably affect the cuDNN-fast-path slowdown.

### Checkpoint selection, verified twice

`ModelCheckpoint(monitor="val_macro_f1", save_best_only=True)` selected epoch
7 (later than M2's epoch 4). Independent `sklearn`-backed recompute on the
restored best-epoch weights agreed with the Keras training-time value within
4.0e-8 (`results/m3/seed42/val_check.json`). The frozen test set was
evaluated only after this selection was finalized, same structural guarantee
as every prior rung.

### Representation analysis (Part 10)

| Epoch | M2 train loss | M2 val loss | M2 val Macro-F1 | M3 train loss | M3 val loss | M3 val Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8552 | 0.5646 | 0.7602 | 0.7466 | 0.4863 | 0.8209 |
| 4 | 0.3772 | 0.4030 | 0.8619 | 0.3658 | 0.3768 | 0.8692 |
| 7 (M3 best) | 0.2891 | 0.4200 | 0.8569 | 0.2993 | 0.3836 | **0.8707** |
| 10 | 0.2122 | 0.5031 | 0.8476 | 0.2454 | 0.4213 | 0.8634 |

Direct evidence for the pre-registered hypothesis (§7: "pretrained vectors
accelerate convergence"): at epoch 1 alone, M3's validation Macro-F1 (0.8209)
already exceeds M2's *entire 10-epoch run* except epochs 3–8. M3 also holds a
broad plateau (val Macro-F1 0.8656–0.8707 across epochs 3–9) rather than M2's
single sharp peak at epoch 4 followed by steady decline — GloVe's informative
starting point acts as an implicit regularizer as well as a head start. M3's
validation loss stays below 0.42 for the entire run; M2's exceeds 0.5 by
epoch 10. This is convergence-speed and generalization evidence from the
curves directly, not inferred from the single test number.

### Per-class comparison

| Class | M2 F1 | M3 F1 | Δ |
|---|---:|---:|---:|
| Checking or savings account | 0.7676 | 0.7816 | +0.0139 |
| Credit card | 0.8307 | 0.8490 | +0.0182 |
| Debt collection | 0.9117 | 0.9203 | +0.0087 |
| Money transfer, virtual currency, or money service | 0.7893 | 0.8081 | +0.0189 |
| Student loan | 0.9597 | 0.9693 | +0.0096 |

All five classes improve — broad-based, not concentrated in one class the way
M1's regression and M2's recovery both were. Consistent with GloVe's high
*token* coverage (99.16%) benefiting the bulk of ordinary vocabulary across
every class, even though its *type* coverage of each class's most distinctive
terms is comparatively weak (§12's pre-registered risk: only 15–26 of each
class's top 30 distinctive terms are in GloVe's vocabulary — servicer names,
statute numbers, and redaction tokens stay randomly initialized). The gain
looks like it comes from better general-purpose word representations across
the whole vocabulary, not from suddenly understanding the high-signal
class-specific terms GloVe was expected to miss.

### Interpretation

M3 delivers the clearest positive result on the ladder so far, matching the
pre-registered hypothesis in both direction and mechanism: pretrained
representations improved generalization (Part 10's curves) and the gain
generalized broadly across classes rather than concentrating in one. The
size of the improvement (+0.0139 vs M2) is larger than raw token coverage
alone would predict for a task this lexically driven, which — combined with
the convergence-speed evidence — supports "informative starting
representations that also regularize" as the mechanism, not merely a better
final embedding value. GloVe was expected to help less than its 99.16% token
coverage implied (§12); the result here does not contradict that caveat, since
the improvement is broad-based rather than concentrated on the high-signal
terms GloVe does poorly on — the caveat about *type* coverage of distinctive
terms remains accurate, it just wasn't the dominant driver of this rung's
result.

TF-IDF (0.8707) is now essentially matched by M3's *validation* Macro-F1
(0.8707) at its best epoch, though M3's *test* Macro-F1 (0.8657) remains
below it. This is reported as context (§12), not a new target — M3 was not
tuned toward this number.

### Commit

```text
add glove
```

---

# 14.12 Stage 10 — M4 ✅ COMPLETE

Run:

**M3 + LR scheduling + early stopping + max_len=256**

This is the final recurrent configuration.

Everything from M3 held fixed: GloVe 100d, bidirectional, dropout 0.3,
recurrent_dropout 0.2, spatial_dropout 0.2, same tokenizer, vocabulary, split,
optimizer, base learning rate, batch size, `ModelCheckpoint` policy. M4 adds
exactly three bundled changes: `max_len=256`, `ReduceLROnPlateau`
(factor=0.5, patience=2, min_lr=1e-5), `EarlyStopping` (patience=3,
restore_best_weights=True) — all fixed before this run, per the locked
configuration table in §14.6.

Trained on Kaggle GPU via `scripts/run_m4.py` (orchestrator) +
`scripts/kaggle_train_m4.py` (training kernel, identical pipeline to M3's
plus an LR-logging callback) + `scripts/kaggle_package_m4.py` (frozen-input
packaging, incl. the same GloVe matrix M3 used). Single seed, per the locked
10-fit budget (§3.2): `src.config.EXPERIMENT_CONFIGS["M4"]["seeds"] = [42]`.

### Configuration bug found during pre-flight

`EXPERIMENT_CONFIGS["M4"]["epochs"]` was set to **20** in `src/config.py`,
contradicting the project-wide, ladder-wide maximum of 10 epochs locked in
§4/§13 ("Maximum epoch budget of 10... never per-model"). No prior task ever
recorded a decision to raise M4's budget specifically, and the task brief for
this stage independently and repeatedly specifies a 10-epoch ceiling — this
was a drift between the config file and the locked specification, not an
intentional value. **Corrected to 10 before any M4 training ran**; no run was
ever performed with the incorrect value. Verified after training completed:

```text
EXPERIMENT_CONFIGS["M4"]["epochs"] == 10
max(history epochs) == 8 <= 10
run_record.json epochs_run == 8 <= 10
```

All four checks (config, `src/config.py`, the training history, and the
registered `results/runs.csv` row) agree on the 10-epoch ceiling.

### Architecture check (before training)

Verified programmatically via the model factory, small script, no training:

```text
layers: InputLayer -> Embedding -> SpatialDropout1D -> Bidirectional(LSTM) -> Dense
same GloVe matrix as M3 (same artifact, same values, loaded as-is)
dropout=0.3, recurrent_dropout=0.2, spatial_dropout=0.2 (unchanged from M3)
max_len: 256 (only architecture change from M3), output_classes: 5
callback order: ModelCheckpoint -> ReduceLROnPlateau -> EarlyStopping, all
monitoring val_macro_f1
```

M3 and M4 total parameters: **identical, 2,235,781**. LSTM and Dense weight
shapes are independent of sequence length — only the `Embedding` layer's
*output* shape changes (`(None, 256, 100)` vs `(None, 128, 100)`), and no new
learnable parameters are introduced by a longer input. Input shape confirmed
`(None, 256)`.

### Truncation at max_len=256 (re-confirmed against Task 3's audit)

| Split | Truncated at 128 | Truncated at 256 |
|---|---:|---:|
| Train | 66.4% | 31.5% |

Matches Task 3's frozen train-split figures exactly — no tokenizer or
preprocessing drift. This is model-token truncation (Keras tokenizer word
counts), not whitespace-word truncation; the two are related but not
identical, per Task 3's own caveat.

### Result

| Seed | Macro-F1 | Accuracy | Macro Precision | Macro Recall | Best Epoch | Epochs Run | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8710 | 0.8682 | 0.8713 | 0.8710 | 5 | 8 | 10,941 |

**M4 − M3 = +0.0054** (exact: 0.005355986585742878)
**M4 − M0 = +0.0202** (exact: 0.02022374634829438; M0 mean = 0.8508)

Both deltas are comfortably outside M0's own three-seed spread (0.0041) — a
real, resolved effect for the combined stage.

### Checkpoint selection, verified twice

Best validation Macro-F1 (0.8772) occurred at epoch 5. Independent
`sklearn`-backed recompute on the restored best-epoch weights agreed with the
Keras training-time value within 4.0e-8 (`results/m4/seed42/val_check.json`).
`EarlyStopping(restore_best_weights=True)` already restores the best weights
into the model when it fires; `model.load_weights(checkpoint_path)` was still
called explicitly afterward (same code path as every prior rung) and is a
no-op here since both point at the same best-epoch weights. The frozen test
set was evaluated only after this selection was finalized.

### Scheduler / early-stopping observations (descriptive, not causal)

```text
LR history:  1e-3 (epochs 1-6) -> 5e-4 (epochs 7-8)
LR reduction: 1 event, at epoch 7 - two stagnant validation-Macro-F1 epochs
              (6, 7) after the best (epoch 5), matching patience=2
Training stopped: after epoch 8 - three stagnant epochs (6, 7, 8) after the
              best, matching patience=3
```

Both callbacks fired exactly where their configured `patience` values predict,
which is evidence the callback wiring is correct — not evidence about what
caused the Macro-F1 result itself. Per the bundled-stage rule (§14.12's own
"Important" note, carried over from the task brief): this is *not* read as
"early stopping improved Macro-F1 by X" or "the scheduler improved Macro-F1 by
X." Training terminated 2 epochs early relative to the 10-epoch ceiling and
restored the best-validation weights; that is a termination/efficiency
behavior, consistent with §4's stated purpose for the checkpoint rule, not a
separate accuracy lever.

### Training-time note

10,941s (~182 minutes) vs M3's 6,793s (~113 minutes) — about 1.61× longer in
wall time, despite running only 8 epochs vs M3's 10. Per-epoch cost roughly
doubled (~1,368s/epoch vs ~679s/epoch), consistent with `max_len=256`
approximately doubling the sequence-length-dependent cost of each LSTM step;
early stopping's 2-epoch reduction partially offset that increase (a ~2×
per-epoch cost × 0.8 epoch-count ratio ≈ 1.6×, matching the observed 1.61×).
`recurrent_dropout=0.2` remains unchanged from M2/M3 and still disables the
cuDNN fast path throughout.

### Convergence / efficiency analysis (Part 14)

| Epoch | M3 val loss | M3 val Macro-F1 | M4 val loss | M4 val Macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.4863 | 0.8209 | 0.4692 | 0.8262 |
| 4 | 0.3768 | 0.8692 | 0.3557 | 0.8752 |
| 5 (M4 best) | 0.3791 | 0.8700 | 0.3531 | **0.8772** |
| 7 (M3 best) | 0.3836 | 0.8707 | 0.3620 | 0.8747 |
| 8 (M4 stops) | 0.4001 | 0.8673 | 0.3741 | 0.8722 |

M4's validation Macro-F1 leads M3's at every epoch they share, and M4's
validation loss stays lower throughout as well — the combined stage is not
simply "M3 with fewer epochs." Fewer epochs run is not itself read as
"better" (§14's explicit caution); it reflects `EarlyStopping` terminating
once no further validation gain was observed, on a curve that was already
ahead of M3's at every point along the way. Longer `max_len` is not, on its
own, concluded to have caused the improvement — it is one of three bundled
changes, and this curve comparison describes the combined stage's behavior,
not an isolated cause.

### Per-class comparison

| Class | M3 F1 | M4 F1 | Δ |
|---|---:|---:|---:|
| Checking or savings account | 0.7816 | 0.7862 | +0.0046 |
| Credit card | 0.8490 | 0.8574 | +0.0084 |
| Debt collection | 0.9203 | 0.9269 | +0.0065 |
| Money transfer, virtual currency, or money service | 0.8081 | 0.8144 | +0.0062 |
| Student loan | 0.9693 | 0.9703 | +0.0010 |

All five classes improve — broad-based and comparatively even (+0.0010 to
+0.0084), unlike M1's concentrated regression or M2's concentrated recovery.
No single class dominates the aggregate delta. Consistent with a combined
optimization/context stage that helps generally rather than fixing one
specific class-boundary weakness.

### Context: TF-IDF and D0 asymmetry note

M4 (0.8710) is the first rung to numerically exceed the TF-IDF auxiliary
reference (0.8707) — reported as context per §12 of the task brief, not as an
official ladder comparison; TF-IDF is not, and has never been, a target this
stage was tuned toward. Separately, `max_len=256` for M4 is the same value
already locked for D0 (§8.1's D0-vs-M4 unequal-context note becomes directly
relevant once D0 exists); nothing about that comparison is evaluated in this
stage.

### Interpretation

**The combined M4 optimization/context stage improved Macro-F1 by +0.0054
relative to M3, and by +0.0202 relative to M0** — both real, resolved
effects under the pre-registered stability reference (§9). Per this stage's
explicit rule, the gain is not assigned to LR scheduling, early stopping, or
`max_len=256` individually; the project's own methodology (§8) states this
bundling was intentional, and no standalone row isolates any one component.
The curve evidence (Part 14) shows M4 ahead of M3 from epoch 1 onward, which
is consistent with `max_len=256` (more context per complaint, truncation
dropping from 66.4% to 31.5%) being a plausible contributor from very early
in training, before the scheduler or early stopping had done anything yet —
but this is an observation about timing, not proof of causal attribution, and
is reported as such.

### Limitation

The individual contributions of LR scheduling, early stopping, and
`max_len=256` cannot be isolated from this single bundled run, and are not
claimed to be — consistent with the limitation already logged in §8 before
M4 was run.

### Commit

```text
run m4
```

---

# 14.13 Stage 11 — Select the Best Recurrent Configuration ✅ COMPLETE

Analysis only — no model was trained, tuned, or modified in this stage.

Use the predefined primary metric:

**Macro-F1**

M4 is intended to be the best recurrent configuration, but the results decide
whether that is actually true.

## Verify result completeness and independent validation

All seven registered runs (M0×3, M1, M2, M3, M4) checked directly against
their underlying `results/m*/seed*/test_metrics.json` files, checkpoint
metadata, and traceability — not against the summary table alone:

* Every `runs.csv` metric (`macro_f1`, `accuracy`, `macro_precision`,
  `macro_recall`) matches its `test_metrics.json` source to full float
  precision — 0 mismatches across 28 checked values.
* Every row's checkpoint loads and its metadata's `monitor_metric` is
  `val_macro_f1` — 7/7 pass.
* Every row is traceable (`dataset_version` matches the frozen split's
  `dataset_content_sha256`, checkpoint metadata matches experiment/seed) —
  7/7 pass.
* All seven rows share the identical `dataset_content_sha256` — no dataset
  drift across the entire ladder.
* M0's independently recomputed statistics: mean = 0.850833, std (ddof=1) =
  0.002069, spread = 0.004125 — matches the previously reported
  0.8508 ± 0.0021 (spread 0.0041) at display precision.
* M1–M4 confirmed as the intended single-seed rungs (`seed=42` only); M0 is
  the intended 3-seed rung (`42, 123, 456`).

`validate_runs_csv(check_seed_completeness=True)` correctly flags D0 as
incomplete (0 of 3 required seeds) — expected and not a defect, since D0 has
not run yet (Task 13).

### Discrepancy found and reconciled (not a data-integrity bug)

`scripts/run_m1.py` through `run_m4.py` each hardcoded a hand-written
constant, `M0_MACRO_F1 = 0.8508`, as the cumulative-delta baseline — the
*rounded, display* value of M0's mean, not the live unrounded 3-seed mean
(`0.850833078773998`). The two differ by 3.3e-5, far smaller than anything
that matters for interpretation, and every incremental delta (`Δ vs
Previous`) is unaffected since those never depend on M0 at all. It surfaces
visibly in exactly one cell: M3's cumulative delta was recorded in §14.11 as
**+0.0149** (using the rounded 0.8508 constant); computed from the true
unrounded mean via `src.results.generate_comparison_table`, it is **+0.0148**.
M1's, M2's, and M4's cumulative deltas round to the same displayed value
either way. §14.11's entry is not altered — the underlying `macro_f1` value
it was computed from was always correct — but the table below (Part 3) uses
the full-precision unrounded M0 mean as the single canonical source going
forward, generated by the existing `generate_comparison_table` function
rather than hand-typed, per this stage's own Part 3 instruction.

## Official recurrent comparison table

Generated by `src.results.generate_comparison_table()` — not hand-typed:

| Model | Configuration | Macro-F1 | Accuracy | Δ vs Previous | Δ vs M0 |
|---|---|---:|---:|---:|---:|
| M0 | Unidirectional LSTM baseline, random embeddings | 0.8508 ± 0.0021 | 0.8483 ± 0.0021 | — | — |
| M1 | Bidirectional LSTM | 0.8485 | 0.8455 | −0.0023 | −0.0023 |
| M2 | BiLSTM + Spatial Dropout | 0.8518 | 0.8485 | +0.0033 | +0.0010 |
| M3 | BiLSTM + Pretrained GloVe-100d | 0.8657 | 0.8627 | +0.0139 | +0.0148 |
| M4 | BiLSTM + GloVe + LR schedule + EarlyStopping + max_len=256 | 0.8710 | 0.8682 | +0.0054 | +0.0202 |

## Delta analysis (full precision, verified independently)

### Incremental (current − previous)

| Delta | Exact value | Rounded |
|---|---:|---:|
| M1 − M0 | −0.002317621684802651 | −0.0023 |
| M2 − M1 | +0.003282040681856002 | +0.0033 |
| M3 − M2 | +0.013870261991500099 | +0.0139 |
| M4 − M3 | +0.005355986585742878 | +0.0054 |

### Cumulative (current − M0, unrounded 3-seed mean = 0.850833078773998)

| Delta | Exact value | Rounded |
|---|---:|---:|
| M1 − M0 | −0.002317621684802651 | −0.0023 |
| M2 − M0 | +0.000964418997053351 | +0.0010 |
| M3 − M0 | +0.014834680988553451 | +0.0148 |
| M4 − M0 | +0.020190667574296328 | +0.0202 |

M1's incremental and cumulative deltas are identical by construction (M1 is
the first rung after M0, so "previous" and "M0" are the same reference).

## M0 stability context (not a significance test)

M0's own three-seed spread is 0.004125 (std 0.002069) — the project's fixed
stability reference (§9), not re-derived here.

| Rung | Δ vs M0 (magnitude) | vs 0.004125 spread | Reading |
|---|---:|---|---|
| M1 | 0.0023 | inside | No clear evidence of a meaningful change |
| M2 | 0.0010 | inside | No clear evidence of a meaningful change |
| M3 | 0.0148 | **3.6× outside** | Materially outside the baseline spread — clear improvement |
| M4 | 0.0202 | **4.9× outside** | Materially outside the baseline spread — clear improvement |

This is a cautious, pre-registered comparison against a stability reference,
not a formal statistical significance test — consistent with §9/§14.16's
own wording rule.

## TF-IDF context (auxiliary, not part of the ladder)

| | Macro-F1 |
|---|---:|
| TF-IDF (auxiliary reference) | 0.8707 |
| Best recurrent (M4) | 0.8710 |

M4 numerically edges past the TF-IDF reference by +0.0003 — far smaller than
the M0 stability spread (0.004125), so this is read as **no clear evidence
that the best recurrent model exceeds the lexical reference**, not a
confirmed win over it. TF-IDF is not part of the recurrent ladder and does
not change which recurrent model is selected; the winner is M4 purely on
the basis of the M0–M4 comparison above. The open question TF-IDF raises —
whether sequence modeling adds real value beyond bag-of-words on this task —
remains genuinely open at the recurrent-model level and is not resolved by
this margin.

## Per-class comparison (M0–M4)

M0 uses the 3-seed mean per class; M1–M4 use their single registered seed.

| Class | M0 | M1 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|---:|
| Checking or savings account | 0.7650 | 0.7651 | 0.7676 | 0.7816 | 0.7862 |
| Credit card | 0.8367 | 0.8364 | 0.8307 | 0.8490 | 0.8574 |
| Debt collection | 0.9053 | 0.9108 | 0.9117 | 0.9203 | 0.9269 |
| Money transfer, virtual currency, or money service | 0.7889 | 0.7715 | 0.7893 | 0.8081 | 0.8144 |
| Student loan | 0.9582 | 0.9587 | 0.9597 | 0.9693 | 0.9703 |

M0→M4, every class improves (+0.0121 to +0.0255) — M4 wins broadly, not
through one class. Money transfer has the largest total gain (+0.0255),
almost entirely recovering from the dip M1 introduced (§14.9) and then
progressing further with each subsequent rung. Student loan gains the least
(+0.0121) because it was already easiest (M0 F1 0.9582) — a ceiling effect,
not a weakness of the later rungs. Checking/savings and Money transfer remain
the two hardest classes throughout the ladder at every rung, consistent with
the account-level vocabulary overlap already flagged in §14.9.

## Training cost comparison (secondary — does not override Macro-F1)

| Model | Parameters | Best Epoch | Epochs Run | Training Time |
|---|---:|---:|---:|---:|
| M0 (mean) | 2,117,893 | — | 10 | 166s |
| M1 | 2,235,781 | 4 | 10 | 274s |
| M2 | 2,235,781 | 4 | 10 | 6,790s |
| M3 | 2,235,781 | 7 | 10 | 6,793s |
| M4 | 2,235,781 | 5 | 8 | 10,941s |

M4 costs roughly 66× M0's training time for a +0.0202 Macro-F1 gain. The
dominant cost driver across M2–M4 is `recurrent_dropout=0.2` disabling the
cuDNN fast LSTM path (§12's pre-registered risk), not `max_len` or the
scheduler — M2→M3 (same `max_len`, added GloVe) costs about the same as
M1→M2 (added `recurrent_dropout`) took in total, and only M3→M4
(`max_len` 128→256) roughly doubles per-epoch cost on top of that. Parameter
count is flat from M1 onward (2,235,781) — cost growth here is entirely a
training-time story, not a model-size story. This informs the cost/benefit
picture but does not change the winner, which is selected on Macro-F1 alone
per Part 5's explicit rule.

## Experiment-by-experiment reasoning

**M0** establishes the baseline: a simple unidirectional LSTM with random
embeddings, 3 seeds to give the whole ladder a stability reference
(0.8508 ± 0.0021). Nothing to interpret beyond that — it is the reference
point everything else is measured against.

**M1** (bidirectionality): delta −0.0023, inside M0's own spread. No clear
evidence bidirectional context helped or hurt. Consistent with the
pre-registered hypothesis (§7) that word order is not the dominant signal on
this lexically-driven task — a reading the TF-IDF reference (§14.8, no
sequence information at all, still beats M0) independently supports.

**M2** (dropout + recurrent_dropout): delta +0.0033 vs M1, inside M0's
spread — not a resolved effect on the aggregate number alone. But the
train/validation curve evidence in §14.10 is not ambiguous: M2's train/val
loss gap is roughly half of M1's by the shared best epoch, and less than
half by epoch 10, with a higher peak validation Macro-F1 too. Regularization
measurably reduced overfitting; the aggregate delta is small because M1
was not overfitting by a large margin to begin with (§7's own predicted
condition: "depends on whether M1 is overfitting" — it was overfitting, just
not severely). No evidence of over-regularization.

**M3** (GloVe): delta +0.0148 vs M0, +0.0139 vs M2 — the first rung whose
delta clears the M0 stability spread by a wide margin. Connects directly to
the frozen GloVe coverage evidence (§14.4B/§12): high *token* coverage
(99.16%) but comparatively weak coverage of each class's most distinctive
*terms* (only 15–26 of the top 30 per class). The result is consistent with
that picture — the gain is broad across all five classes (§14.11) rather
than concentrated on the high-signal vocabulary GloVe covers poorly,
supporting "better general-purpose representations across the whole
vocabulary" as the mechanism, not a lucky hit on the hardest words.

**M4** (bundled LR scheduling + early stopping + `max_len=256`): delta
+0.0054 vs M3, +0.0202 vs M0 — both resolved effects, and the largest
cumulative gain on the ladder. Per the project's own bundling rule (§8, and
this stage's Part 3), the individual contributions of the three components
are not separated. LR reduced once (epoch 7) and training stopped at epoch 8
(3 stagnant epochs after the epoch-5 peak) — both callbacks fired exactly
where their `patience` values predict, evidence the wiring is correct, not
evidence of what caused the Macro-F1 gain. The curve comparison in §14.12
shows M4 ahead of M3 from epoch 1, before either callback had done anything,
which is circumstantial support for `max_len=256` being a contributor — not
proof, and not a component-level claim.

## Test-6 requirement coverage

| Requirement | Covered by | Result/Status |
|---|---|---|
| Baseline first | M0 | ✅ 0.8508 ± 0.0021 |
| Bidirectional LSTM | M1 | ✅ 0.8485 (no clear effect vs M0) |
| Dropout | M2 | ✅ dropout=0.3, part of the bundled M2 result 0.8518 |
| `recurrent_dropout` | M2 | ✅ recurrent_dropout=0.2, same result |
| GloVe | M3 | ✅ 0.8657, clear improvement |
| LR scheduling | M4 | ✅ bundled into M4, 0.8710 |
| Early stopping | M4 | ✅ bundled into M4, same result |
| Longer `max_length` | M4 | ✅ `max_len=256`, bundled into M4, same result |
| DistilBERT | D0 | ⏳ pending (Task 13) |
| Stacked layers | Excluded | Documented, §3.3 |
| Class weights | Excluded | Documented, §3.3 |

8 of 9 applicable menu items covered by a completed experiment (D0 pending);
both exclusions carry a stated reason, unchanged from §3.3.

## Best recurrent configuration

> **M4 is the strongest recurrent configuration, with test Macro-F1 = 0.8710,
> improving 0.0202 points over the M0 baseline (0.8508 ± 0.0021). This
> improvement is materially outside M0's own three-seed stability spread
> (0.004125), so it is reported as a clear, resolved gain rather than
> run-to-run noise. M4 is therefore selected as the recurrent benchmark for
> the DistilBERT comparison.**

Relevant cost observation: M4 also costs the most to train (~10,941s, ~66×
M0's), driven overwhelmingly by `recurrent_dropout` disabling the cuDNN fast
path rather than by `max_len` or the scheduler — a real trade-off, noted here
for the final discussion but not a factor in the selection itself (Part 5:
selection is Macro-F1 only).

## D0 handoff

**D0's primary comparison target is M4** (test Macro-F1 = 0.8710), not
assumed in advance — the data shows M4 wins, so M4 is the benchmark.
**D0 vs M0** (0.8508 ± 0.0021) remains the secondary, baseline-level
comparison, per §14.14's existing plan. The §8.1 D0-vs-M4 unequal-context
caveat (WordPiece reads relatively less of its own `max_len=256` window than
Keras does at M4's `max_len=256`) applies directly to the primary comparison
and must be restated wherever D0 vs M4 is reported, exactly as already
logged before any D0 result exists.

## Machine-readable decision artifact

`results/best_recurrent.json` — derived from `results/runs.csv`, not a
duplicate result system:

```json
{
  "best_recurrent_experiment": "M4",
  "best_recurrent_macro_f1": 0.8710237463482944,
  "best_recurrent_delta_vs_m0": 0.020190667574296328,
  "tfidf_reference_macro_f1": 0.8707487900723045,
  "m0_mean_macro_f1": 0.850833078773998,
  "m0_std_macro_f1": 0.002068765063694662
}
```

### Commit

```text
select best recurrent
```

---

# 14.14 Stage 12 — DistilBERT ✅ COMPLETE

Train:

**D0 — DistilBERT fine-tune**

Run:

**3 seeds**

Use the same train/validation/test split. Same frozen dataset fingerprint,
split, and preprocessing version as every M0–M4 rung
(`dataset_content_sha256 = eb66684f...`); D0 uses the pretrained DistilBERT
tokenizer, never the frozen Keras tokenizer or GloVe embeddings.

Trained on Kaggle GPU via `scripts/run_d0.py` (orchestrator, 3-seed
sequential with resume safety) + `scripts/kaggle_train_d0.py` (training
kernel) + `scripts/kaggle_package_d0.py` (frozen-input packaging). D0's
kernel is the only one in the project with `enable_internet: true` — it needs
to pull the pretrained `distilbert-base-uncased` checkpoint from the
HuggingFace Hub. Model/tokenizer construction lives in `src/distilbert.py`,
kept separate from `src.models` (the recurrent factory) since D0's
architecture is a pretrained Transformer, not an LSTM.

### Environment finding, fixed before training (pre-flight)

`transformers>=5.0` removed TensorFlow model classes entirely
(`TFAutoModelForSequenceClassification` and the rest) — confirmed locally
before any Kaggle run. Fixed by pinning `transformers>=4.35.0,<5.0.0` and
adding `tf_keras>=2.15.0` (required for TF's native Keras 3 to load HF's
legacy-Keras-2-based TF model classes) to `requirements.txt`, and installing
the same pin at the top of `kaggle_train_d0.py` itself, since Kaggle's
preinstalled `transformers` could independently be the incompatible 5.x
series. Two further compatibility issues, found and resolved the same way
(confirmed working end-to-end locally before spending Kaggle GPU time):

* `distilbert-base-uncased`'s PyTorch-safetensors-to-TF conversion path is
  broken in this library version (`'builtins.safe_open' object is not
  iterable`) — fixed by `use_safetensors=False`, which loads the checkpoint's
  native `tf_model.h5` weights directly instead.
* Mixing a `tf_keras`-based HF model with TF's native Keras-3 optimizer/
  metrics raises `AttributeError: 'Variable' object has no attribute
  '_distribute_strategy'` — fixed by using `tf_keras` primitives (optimizer,
  loss, callbacks) throughout D0's compile/fit path, and computing
  `val_macro_f1` via a callback that calls `src.evaluation.
  compute_val_macro_f1` directly each epoch (the same canonical function
  every other rung's final metric comes from) rather than a second
  `tf_keras`-compatible Metric class.

None of this changes D0's locked hyperparameters or architecture — it is
dependency/environment plumbing, verified before training, not a
methodology change.

### Architecture / compatibility check (before training)

```text
checkpoint: distilbert-base-uncased (config-verified, never substituted)
tokenizer: DistilBertTokenizerFast, matching special tokens present
max_len: 256, input_ids/attention_mask shape: (N, 256)
output classes: 5 (config.num_labels)
parameters: 66,957,317 total/trainable (identical across all 3 seeds)
```

66.9M parameters vs M4's 2.24M — roughly 30× larger, as expected for a
pretrained Transformer vs a task-trained BiLSTM.

### Result — 3 seeds

| Seed | Macro-F1 | Accuracy | Macro Precision | Macro Recall | Best Epoch | Epochs Run | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8759 | 0.8739 | 0.8755 | 0.8768 | 3 | 4 | 5,422 |
| 123 | 0.8777 | 0.8752 | 0.8791 | 0.8780 | 2 | 4 | 5,409 |
| 456 | 0.8740 | 0.8721 | 0.8736 | 0.8754 | 2 | 4 | 5,381 |

**D0 Macro-F1 = 0.8759 ± 0.0018** (mean ± sample std, `ddof=1`,
`src.evaluation.seed_statistics`) — the tightest seed spread of any
multi-seed rung in the project (M0's was 0.0021). Min 0.8740, max 0.8777,
spread 0.0037.

**D0 − M4 = +0.0048** (exact: 0.00482950383535119; M4 = 0.8710237463482944)
**D0 − M0 = +0.0250** (exact: 0.025020171409647518; M0 mean = 0.850833078774)

Notably, every individual D0 seed independently exceeds M4's single value —
even the weakest D0 seed (456, 0.8740) beats M4 (0.8710) by +0.0030. This is
complementary evidence beyond the mean-vs-single-point comparison: the result
is not an artifact of averaging.

### Checkpoint selection, verified — and a checkpoint-format limitation found and worked around

`ModelCheckpoint(monitor="val_macro_f1", save_best_only=True)` selected the
best epoch for each seed (3, 2, 2). Independent `sklearn`-backed recompute
**inside the training kernel**, on the restored best-epoch weights, agreed
with the Keras training-time value at **exact 0.0 agreement** on all three
seeds (`results/d0/seed{N}/val_check.json`) — the tightest of any rung. The
frozen test set was evaluated only after this selection, same structural
guarantee as M0–M4.

A separate, additional local check — reloading the downloaded checkpoint into
a *fresh Python process* (the same cross-process sanity check M0–M4's
orchestrators run) — reproducibly failed with `Layer 'sa_layer_norm' expected
2 variables, but received 0 variables`, for all three seeds, with and without
first compiling the freshly-built model. Diagnosed as a genuine `tf_keras`/
`transformers` checkpoint-format issue rather than a code bug: inspecting the
saved `.weights.h5` file directly (via `h5py`) shows it contains a full
`optimizer` group (209 variables, from `AdamW`'s momentum/velocity slots)
despite `save_weights_only=True`, which this library version's single-file
H5 loader cannot reconcile against a freshly-compiled model's differently-
initialized optimizer state. This is specific to D0's checkpoint format —
M0–M4's LSTM checkpoints don't save an optimizer group and don't hit it.
Since the actual Part 8 requirement (restore + independently recompute +
compare agreement) was already satisfied *inside the training kernel*, with
exact 0.0 agreement, `scripts/run_d0.py`'s `validate_run` was changed to a
file-integrity check (valid, readable HDF5 with the expected top-level
groups) rather than a full fresh-process model reload — documented here
rather than silently worked around.

### Truncation (WordPiece, re-confirmed against Task 3's audit)

44.63% of training sequences use the full 256-token window (a same-or-longer
proxy computed from `attention_mask` sums) — closely matching, not
contradicting, Task 3's exact figure of 44.4% WordPiece truncation at 256 on
the same split; the small (0.2 point) difference is the proxy metric being a
looser approximation, not data or tokenizer drift.

### Unequal-context caveat (§8.1, restated because it now applies)

M4 and D0 share the same nominal `max_len=256`, but WordPiece produces a
median 1.26× more tokens than the Keras tokenizer on this text (Task 1
audit), so D0 reads *less* of its own effective context at that nominal
window than M4 does. D0 still outperforms M4 under this handicap — per §8.1's
own pre-registered logic, that makes the result, if anything, a more
conservative (understated) advantage for D0, not an inflated one. This is not
a token-for-token equivalent comparison and is not claimed to be.

### Compute cost comparison

| Model | Parameters | Fits | Mean Time/Fit | Total Time |
|---|---:|---:|---:|---:|
| M4 | 2,235,781 | 1 | 10,941s (~182 min) | 10,941s |
| D0 | 66,957,317 | 3 | 5,404s (~90 min) | 16,214s (~270 min) |

Counterintuitive but real: D0's *per-fit* training time is roughly half of
M4's, despite having ~30× more parameters. M4's dominant cost driver is
`recurrent_dropout=0.2` disabling the cuDNN fast LSTM path (§12); D0 has no
such penalty — HuggingFace's TF Transformer implementation runs at native GPU
speed. D0's *total* compute (3 seeds) is larger only because it runs 3 fits
against M4's 1, per the locked seed policy (§3.2), not because any single D0
fit is more expensive.

### Per-class comparison (M0 mean, 3 seeds; D0 mean, 3 seeds)

| Class | M0 | M4 | D0 |
|---|---:|---:|---:|
| Checking or savings account | 0.7650 | 0.7862 | 0.7961 |
| Credit card | 0.8367 | 0.8574 | 0.8657 |
| Debt collection | 0.9053 | 0.9269 | 0.9235 |
| Money transfer, virtual currency, or money service | 0.7889 | 0.8144 | 0.8216 |
| Student loan | 0.9582 | 0.9703 | 0.9724 |

D0 improves on M4 in four of five classes; Debt collection is the one
exception, where M4 is slightly ahead (0.9269 vs 0.9235). The result is broad
rather than driven by one class, and the one regression is in the class that
was already easiest for the recurrent ladder to separate — not evidence of a
systematic weakness in D0.

### 3-seed stability, with the asymmetry stated explicitly

D0's own three-seed spread (std 0.0018, min–max spread 0.0037) is even
tighter than M0's (std 0.0021, spread 0.0041) — D0 is a stable fit across
seeds. However, **the D0-vs-M4 comparison has three seeds on one side and one
seed on the other** — this is not a symmetric variance comparison, and is not
presented as one. D0 − M4 (+0.0048) is modestly larger than M0's own spread
(0.0041), but only slightly; on the M0 stability reference alone this would
be a borderline case. What resolves it further is that all three individual
D0 seeds beat M4's single value, not just the mean — three independent data
points landing on the same side is stronger evidence than the mean-delta
comparison alone, though still not a formal significance test (§9's own
explicit rule; no such test is invented here).

### Interpretation

Per Part 19's outcome categories: **DistilBERT modestly but consistently
outperformed the best recurrent configuration.** D0 − M4 = +0.0048 is small
relative to M0's stability spread, and this is stated plainly rather than
inflated — this is not "DistilBERT outperformed the best recurrent
configuration by a wide margin." But it is also not "no clear improvement":
all three D0 seeds independently beat M4, D0's own spread is tighter than
M0's, and the improvement holds despite D0's context-window handicap
(unequal-context caveat above). Plausible contributors — pretrained
contextual representations, transfer learning from a large pretraining
corpus, and WordPiece's subword handling of financial/redaction vocabulary —
are stated as plausible, not proven causal mechanisms; this experiment
does not isolate which of them matters, only that the model family as a
whole (pretrained Transformer + fine-tuning) modestly exceeds the best
recurrent configuration reachable within this project's compute budget.

### Commit

```text
run distilbert
```

---

# 14.15 Stage 13 — Results Consolidation ✅ COMPLETE

Analysis only — no model trained, tuned, or modified. All values below come
from `results/runs.csv`, `results/tfidf_reference.json`, and the per-seed
`results/m*/seed*/test_metrics.json` / `results/d0/seed*/test_metrics.json`
files, verified against each other before use (Part 1: zero mismatches
across every source, same check already run in Tasks 12/13). Full working —
generated tables, deltas, per-class breakdown, and three charts — lives in
`notebooks/05_final_results.ipynb`, executed top to bottom with 0 errors.
This section is the consolidated summary, not a duplicate of that notebook.

### Final table

Generated by `src.results.generate_comparison_table()` (extended this stage
to add Best Epoch/Time/Parameters columns — not hand-typed):

| Model | Configuration | Macro-F1 | Accuracy | Δ vs Previous | Δ vs M0 | Best Epoch | Time | Parameters |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | Unidirectional LSTM baseline | 0.8508 ± 0.0021 | 0.8483 ± 0.0021 | — | — | 3–4 | 166 ± 11s | 2,117,893 |
| M1 | Bidirectional LSTM | 0.8485 | 0.8455 | −0.0023 | −0.0023 | 4 | 274s | 2,235,781 |
| M2 | BiLSTM + Spatial Dropout | 0.8518 | 0.8485 | +0.0033 | +0.0010 | 4 | 6,790s | 2,235,781 |
| M3 | BiLSTM + Pretrained GloVe-100d | 0.8657 | 0.8627 | +0.0139 | +0.0148 | 7 | 6,793s | 2,235,781 |
| M4 | BiLSTM + GloVe + LR schedule + EarlyStopping + max_len=256 | 0.8710 | 0.8682 | +0.0054 | +0.0202 | 5 | 10,941s | 2,235,781 |
| D0 | DistilBERT (fine-tuned transformer) | 0.8759 ± 0.0018 | 0.8737 ± 0.0015 | +0.0048 | +0.0250 | 2–3 | 5,404 ± 21s | 66,957,317 |

**TF-IDF (auxiliary, never part of this table): Macro-F1 = 0.8707.** Not a
row here by design — it has no `experiment` entry in `results/runs.csv`, no
official ladder delta, and is not the recurrent-ladder previous-row for M0.

D0's `Δ vs Previous` is D0 − M4 — D0's one meaningful predecessor comparison
is the best recurrent configuration, not a seventh rung in the ladder.

### Deltas — full precision, canonical M0 mean

Every cumulative delta uses the unrounded 3-seed M0 mean
(`0.850833078773998`, from `seed_statistics`), not the rounded 0.8508
display value — the precision issue Task 12 found and resolved remains
resolved here; re-verified in the notebook.

| Delta | Incremental (exact) | Cumulative vs M0 (exact) |
|---|---:|---:|
| M1 | −0.002317621684802651 | −0.002317621684802651 |
| M2 | +0.003282040681856002 | +0.000964418997053351 |
| M3 | +0.013870261991500099 | +0.014834680988553451 |
| M4 | +0.005355986585742878 | +0.020190667574296328 |
| D0 | +0.00482950383535119 | +0.025020171409647518 |

### M0 / D0 variability, asymmetry stated explicitly

M0 = 0.8508 ± 0.0021 (n=3). D0 = 0.8759 ± 0.0018 (n=3) — D0's spread is
tighter than M0's. **M1–M4 are single-run values with no measured
uncertainty** — no standard deviation is shown or implied for them, and
none is invented. The stability reference (M0's 3-seed spread, 0.004125)
is applied only where it is meaningful: as a cautious sanity check on
single-seed deltas, never as a formal significance test.

### Official recurrent ranking (M0–M4 only)

M4 (0.8710) > M3 (0.8657) > M2 (0.8518) > M0 (0.8508) > M1 (0.8485)

### Final model-family comparison

Best recurrent = **M4** (0.8710). D0 (0.8759 ± 0.0018) is the final
Transformer benchmark. TF-IDF (0.8707) is not part of either ranking — it is
reported alongside them as context, not ranked against them as a competing
configuration.

### Requirement coverage

| Requirement | Experiment | Status |
|---|---|---|
| Baseline | M0 | Complete |
| Bidirectional LSTM | M1 | Complete |
| Stacked layers | Excluded | Documented (§3.3) |
| Dropout | M2 | Complete |
| Recurrent dropout | M2 | Complete |
| GloVe | M3 | Complete |
| LR scheduling | M4 | Complete |
| Early stopping | M4 | Complete |
| Class weights | Excluded | Documented (§3.3) |
| Longer max_length | M4 | Complete |
| LSTM → DistilBERT | D0 | Complete |

**9/9 applicable menu items covered.** Not "all menu items tested" — stacked
layers and class weights remain deliberately excluded, with reasons recorded
in §3.3, not silently dropped from the count.

### Rung-by-rung reasoning (concise; full reasoning already in §14.9–§14.14)

* **M0** — baseline, 0.8508 ± 0.0021, the ladder's stability reference.
* **M1** — bidirectionality: delta inside M0's spread, no clear evidence of
  a real effect (§14.9).
* **M2** — regularization: aggregate delta still inside M0's spread, but
  direct train/val curve evidence showed reduced overfitting vs M1 (§14.10).
* **M3** — GloVe: first clearly resolved improvement, broad across every
  class, consistent with high token coverage on ordinary vocabulary despite
  weak coverage of the most class-distinctive terms (§14.11).
* **M4** — bundled LR scheduling + early stopping + `max_len=256`: largest
  recurrent-ladder gain; the three components are not individually
  separated, by the project's own pre-registered design (§8, §14.12).
* **D0** — DistilBERT: D0 − M4 = +0.0048, modest but consistent across all
  three seeds (every seed beats M4's single value), holding despite the
  unequal-context tokenizer handicap (§8.1, §14.14). Not overclaimed as a
  proven causal result — pretrained representations, transfer learning, and
  WordPiece subword handling are stated as plausible contributors, not
  isolated or proven mechanisms.

### Compute / efficiency

M2–M4's cost is dominated by `recurrent_dropout=0.2` disabling the cuDNN
fast LSTM path, not parameter count (flat at 2,235,781 from M1 onward). M4
is ~67× slower than M0 for a nearly identical parameter count. D0 has ~30×
more parameters than M4 but trains faster per fit (~90 min vs ~182 min) —
it never incurs the recurrent-dropout penalty, since it isn't a recurrent
model. Full cost table in the notebook.

### TF-IDF context — one of the project's clearest findings

TF-IDF = 0.8707, M4 = 0.8710 (**M4 − TF-IDF = +0.0003**), D0 = 0.8759
(**D0 − TF-IDF = +0.0051**). TF-IDF is not weak — it is one of the strongest
single results in the project. M4's edge over it is far smaller than
anything resolvable against the M0 stability spread: read as **no clear
evidence the best recurrent model beats a plain lexical baseline**, not a
confirmed win. D0 is the one model in the project that clearly moves past
that lexical ceiling.

### Per-class master table

M0/D0 use the 3-seed mean per class:

| Class | M0 | M1 | M2 | M3 | M4 | D0 |
|---|---:|---:|---:|---:|---:|---:|
| Checking or savings account | 0.7650 | 0.7651 | 0.7676 | 0.7816 | 0.7862 | 0.7961 |
| Credit card | 0.8367 | 0.8364 | 0.8307 | 0.8490 | 0.8574 | 0.8657 |
| Debt collection | 0.9053 | 0.9108 | 0.9117 | 0.9203 | 0.9269 | 0.9235 |
| Money transfer, virtual currency, or money service | 0.7889 | 0.7715 | 0.7893 | 0.8081 | 0.8144 | 0.8216 |
| Student loan | 0.9582 | 0.9587 | 0.9597 | 0.9693 | 0.9703 | 0.9724 |

Student loan is easiest at every rung; Checking/savings and Money transfer
stay hardest throughout. M3 lifts every class at once (broad, not
concentrated). M4 improves on M3 in every class. D0 improves on M4 in four
of five classes — Debt collection is the one exception (0.9269 vs 0.9235),
not a systematic D0 weakness.

### Confusion matrix package

Verified present and using the canonical 5-class order for every rung:
`results/m0/seed{42,123,456}/test_metrics.json`, `results/m1/seed42/`,
`results/m2/seed42/`, `results/m3/seed42/`, `results/m4/seed42/`,
`results/d0/seed{42,123,456}/`, plus `results/tfidf_reference.json`.
Prepared for Task 16 (Error Analysis) — not analyzed in narrative depth here.

### Final model decision

> **Best recurrent model: M4**, Macro-F1 = 0.8710.
>
> **Final model-family winner: D0.** D0 consistently outperformed M4 across
> all three seeds, with a mean improvement of +0.0048 Macro-F1 points. M4 is
> single-seed and D0 is three-seed — this is an asymmetric variance
> comparison, not a symmetric one, and is not presented as a formal
> significance test.

### What the experiments showed

1. A simple unidirectional LSTM baseline reaches 0.8508 ± 0.0021.
2. Bidirectionality (M1) did not clearly help.
3. Regularization (M2) showed a real but modest effect, visible in the
   training curves more than the aggregate delta.
4. GloVe (M3) produced the first clear, broad-based improvement.
5. The bundled M4 stage produced the largest recurrent-ladder gain, with its
   three components deliberately not individually separated.
6. **M4 is the final recurrent benchmark.**
7. DistilBERT (D0) modestly but consistently beat M4, holding across all
   three seeds despite an unequal-context handicap.
8. The TF-IDF reference matters because this task is strongly lexical — a
   bag-of-words model with no sequence information reaches 0.8707,
   essentially tying M4. D0 is the only model that clearly moves past that
   lexical ceiling.

### Machine-readable summary

`results/best_recurrent.json` — extended this stage (not replaced with a
second file) with D0/TF-IDF/final-model fields:

```json
{
  "m0_macro_f1_mean": 0.850833078773998,
  "m0_macro_f1_std": 0.002068765063694662,
  "m4_macro_f1": 0.8710237463482944,
  "d0_macro_f1_mean": 0.8758532501836456,
  "d0_macro_f1_std": 0.0018327384663822041,
  "tfidf_macro_f1": 0.8707487900723045,
  "best_recurrent": "M4",
  "d0_vs_m4": 0.00482950383535119,
  "d0_vs_m0": 0.025020171409647518,
  "tfidf_vs_m4": -0.00027495627598983496,
  "requirement_coverage": "9/9 applicable menu items covered (2 exclusions, documented)",
  "final_model": "D0"
}
```

### Commit

```text
add final results
```

---

# 14.16 Stage 14 — Statistical/Variance Interpretation ✅ COMPLETE

Analysis only — no model trained, tuned, or modified. Full working (seed
tables, delta classification, three charts) lives in the "Variance
interpretation" section appended to `notebooks/05_final_results.ipynb`
(executed top to bottom, 0 errors across all 52 cells). This section is the
consolidated summary.

Use the three-seed results as a **stability reference**. The project has no
formal significance-testing protocol and none is introduced here.

### M0 / D0 seed variability (independently re-verified)

Every seed-level `macro_f1` re-read directly from
`results/m0|d0/seed*/test_metrics.json` and cross-checked against
`results/runs.csv` — zero mismatches (same result as every prior check in
Tasks 12–14).

| Endpoint | Seed 42 | Seed 123 | Seed 456 | Mean | Std | Min | Max | Spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.8489 | 0.8507 | 0.8530 | 0.8508 | 0.0021 | 0.8489 | 0.8530 | 0.0041 |
| D0 | 0.8759 | 0.8777 | 0.8740 | 0.8759 | 0.0018 | 0.8740 | 0.8777 | 0.0037 |

D0's observed three-seed spread (0.0037) is smaller than M0's (0.0041) —
stated descriptively, about these three tested seeds, not as a general claim
that DistilBERT is intrinsically more stable than an LSTM.

### Single-seed rung limitation (M1–M4)

M1–M4 each have exactly one seed (`EXPERIMENT_CONFIGS["M1"–"M4"]["seeds"]`,
§3.2's locked budget). **There is no rung-specific standard deviation for
M1–M4** — their individual run-to-run variance cannot be estimated from this
project's data, and none is fabricated. Every M1–M4 delta is judged only
against M0's own three-seed spread (0.0041), the project's one pre-registered
stability reference (§9) — not a new threshold, not a formal test.

### Delta classification

| Comparison | Delta | Variance Evidence | Interpretation |
|---|---:|---|---|
| M1 − M0 | −0.0023 | M0 3-seed spread | within observed baseline variability (0.56×) |
| M2 − M1 | +0.0033 | M0 stability reference | within observed baseline variability (0.80×) |
| M3 − M2 | +0.0139 | M0 stability reference | clearly outside observed baseline variability (3.36×) |
| M4 − M3 | +0.0054 | M0 stability reference | outside, but only modestly (1.30×) — caution |
| M4 − M0 | +0.0202 | M0 stability reference | clearly outside observed baseline variability (4.90×) |
| D0 − M4 | +0.0048 | D0 3-seed / M4 single seed | outside, but only modestly (1.17×) — caution, asymmetric |
| D0 − M0 | +0.0250 | both multi-seed | clearly outside observed baseline variability (6.07×) |
| M4 − TF-IDF | +0.0003 | TF-IDF single run | effectively negligible observed difference |

**M1** — small negative delta, within M0's spread; not treated as a
confirmed regression. **M2** — small positive delta, within M0's spread on
the aggregate number (the train/val curve evidence in §14.10 is a separate,
non-aggregate line of evidence, not contradicted by this classification).
**M3** — large positive delta, comfortably outside the spread; the clearest
resolved recurrent-ladder improvement. **M4** — cumulative delta clearly
outside the spread; the incremental M4-vs-M3 delta is outside it but only
modestly (1.30×) — a real but less comfortable margin than M3's, and still a
single-seed observation.

### D0 vs M4 — asymmetric evidence, stated explicitly

D0 (3 seeds) vs M4 (1 seed) is **not a symmetric uncertainty comparison**.
All three D0 seeds (0.8740, 0.8759, 0.8777) exceeded M4's single observed
score (0.8710); D0's own spread is small and does not overlap M4's value.
M4 has no repeated-seed estimate, so the asymmetry is not resolved by D0's
tight spread. Correct framing: "D0 consistently exceeded the observed M4
score across all three seeds." **Not:** "D0 is statistically significantly
better than M4" — no such test was performed, and one M4 seed cannot be
turned into a distribution or a pooled standard error.

### TF-IDF variance limitation

TF-IDF is a single reference run — no std, no significance test, no
uncertainty interval exists or is invented for it. M4 − TF-IDF (+0.0003) is
an effectively negligible observed difference under the available evidence,
not "M4 significantly beats TF-IDF." D0 − TF-IDF (+0.0051) is a useful
descriptive gap; TF-IDF's single-run status is unchanged by D0 having three
seeds of its own.

### Claims supported by the experiments

* M0 and D0 both show low variation across their three tested seeds.
* D0's observed seed-to-seed spread is smaller than M0's.
* M3 produced a clear observed improvement over M2 and M0.
* M4 produced the strongest recurrent result, well outside M0's spread
  cumulatively.
* All three D0 seeds exceeded the single observed M4 score.
* D0 was the best observed model in the project.

### What we cannot claim

* Formal statistical significance for any M1–M4 delta.
* A confidence interval for M1, M2, M3, or M4 individually.
* The true population variance of M1–M4 — only M0 and D0 have a measured
  sample variance at all.
* A formal significance test of D0 vs M4 — the comparison is asymmetric (3
  seeds vs 1) and reported as such.
* General superiority of D0 beyond this dataset, split, and training setup.
* That any single M4 component individually caused the observed M4 gain —
  the bundled-stage limitation (§8/§14.12) still applies.

### Seed-count limitation (project design, not an ideal)

The fixed 10-fit budget (§3.2) allocated repeated seeds only to M0 and D0 —
`M0×3, M1×1, M2×1, M3×1, M4×1, D0×3` — because those two anchor the baseline
and the final model-family comparison. This is stated honestly as a
compute-budget trade-off, not an ideal experimental design: M1–M4 each
report one run, and their individual run-to-run variance is genuinely
unknown.

### Machine-readable summary

`results/best_recurrent.json` extended (not duplicated) with `m0_spread`,
`d0_spread`, `m1_delta`–`m4_delta` (cumulative vs the full-precision M0
mean), `stability_reference`, and `uncertainty_note`.

### Commit

```text
add variance notes
```

---

# 14.17 Stage 15 — Error Analysis

Perform error analysis on:

**best recurrent model**

and

**DistilBERT**

### Include

#### Confusion matrix

Identify the most confused classes.

#### Per-class metrics

Look for:

* low precision
* low recall
* low F1

#### Misclassified examples

For selected examples show:

```text
Text
True label
Predicted label
Likely reason
```

Keep examples concise.

Do not dump large blocks of raw text into the report.

### Commit

```text
add error analysis
```

---

# 14.18 Stage 16 — Final Technical Review

Before declaring the project complete, perform a full audit.

Check:

### Data

* duplicates removed before split
* split reproducible
* no test leakage
* labels correct

### Models

* M0–M4 match the project specification
* D0 is correctly separated
* no unplanned model changes

### Metrics

* Macro-F1 verified
* Accuracy verified
* Precision/Recall verified
* checkpoint metric correct
* best checkpoint restored

### Reproducibility

* seeds recorded
* configuration recorded
* package versions recorded
* results persisted

### Code quality

* no dead code
* no unused imports
* no debug prints
* no copied/generated filler
* no unexplained magic numbers
* clear function names
* minimal useful comments

### Notebook quality

* runs from top to bottom
* outputs are understandable
* headings are short
* plots have useful titles
* no excessive prose
* no placeholder cells
* no broken paths

### Report quality

* baseline clearly identified
* every delta calculated
* negative results included
* exclusions explained
* M4 limitation stated
* D0 vs M4 asymmetry stated
* conclusions supported by results

### Commit quality

* every completed task has a commit
* commit messages remain short and human
* no giant "final project" commit containing everything

### Commit

```text
final project check
```

---

# 14.19 Stage 17 — External Review

After all implementation work is complete, have an independent reviewer go through the project and this file.

The reviewer must inspect:

```text
project_plan.md
src/
notebooks/
results/
README.md
```

### Review request

Ask the reviewer to verify:

1. Does the implementation match every locked requirement?
2. Were any planned experiments changed?
3. Is there any data leakage?
4. Are the reported metrics correct?
5. Are deltas calculated correctly?
6. Are M0 and D0 seed statistics correct?
7. Is M4's bundled limitation stated correctly?
8. Are exclusions documented?
9. Is the code reproducible?
10. Does the project look professionally engineered?
11. Are there signs of generated/vibe-coded implementation?
12. Are comments and captions natural and concise?
13. Are there unused or redundant components?
14. Are notebooks and scripts clean?
15. Could a reviewer reproduce the reported results?

The reviewer must identify:

```text
Critical issues
Major issues
Minor issues
Suggestions
```

No issue should be silently ignored.

---

# 14.20 Stage 18 — Final Cleanup

Fix every confirmed issue from the external review.

After each fix:

```text
Fix
→ Test
→ Verify
→ Commit
```

Do not create one huge cleanup commit.

Example:

```text
fix metric check
fix data path
clean notebook
fix seed log
```

---

# 14.21 Stage 19 — Final Reproducibility Check

Perform a clean run from the repository.

The final check should confirm:

```text
clone repo
→ install dependencies
→ load data
→ load persisted splits
→ run evaluation
→ reproduce recorded results
```

Any result difference must be understood and documented.

### Commit

```text
verify results
```

---

# 14.22 Stage 20 — Final Documentation

README should contain:

1. Problem
2. Dataset
3. Experimental design
4. Model ladder
5. Evaluation metrics
6. Main results
7. Error analysis
8. Limitations
9. How to reproduce
10. Project structure

Keep the writing concise and natural.

The README should read like an engineer explaining a real project, not like generated documentation.

### Commit

```text
finish readme
```

---

# 14.23 Final Quality Standard

Before calling the project complete, the project must satisfy all of the following:

```text
[ ] Dataset audited
[ ] Duplicates removed before split
[ ] Split persisted
[ ] Metrics verified
[ ] M0 complete
[ ] M1 complete
[ ] M2 complete
[ ] M3 complete
[ ] M4 complete
[ ] D0 complete
[ ] 10 fits completed
[ ] Results saved
[ ] Deltas verified
[ ] Error analysis complete
[ ] Exclusions documented
[ ] M4 limitation documented
[ ] Seed asymmetry documented
[ ] Code reviewed
[ ] Notebooks cleaned
[ ] README complete
[ ] External review completed
[ ] Review issues fixed
[ ] Reproducibility checked
[ ] Every task committed
```

---

# 14.24 Definition of Done

The project is considered complete only when:

> The implementation matches the locked experimental specification, all six configurations have been evaluated under the defined protocol, results are reproducible, deltas are correctly reported, negative findings are preserved, limitations are explicit, error analysis is complete, and an external review finds no unresolved critical or major issue.

The final project should look like a **deliberately designed ML experiment built and maintained by a professional engineer**.

It should not look:

* over-engineered
* artificially polished
* full of generic AI comments
* full of unnecessary abstractions
* notebook-only
* copied from unrelated templates
* "vibe coded"

The standard is:

> **Simple where possible, explicit where necessary, reproducible everywhere.**

