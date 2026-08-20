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

# 14.7 Stage 5 — M0 Baseline

Train:

**Simple LSTM**

Configuration:

```text
random embeddings
max_len=128
```

Run:

**3 seeds**

### Tasks

* train
* checkpoint best validation model
* record metrics
* record training time
* record parameter count
* record best epoch
* calculate mean ± std

### Required output

M0 becomes the official baseline.

### Gate

Inspect M0 variance before continuing.

If the baseline is unstable, investigate before running the ladder.

### Commit

```text
add baseline
```

---

# 14.8 Stage 6 — Auxiliary TF-IDF Reference

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

### Commit

```text
add tfidf check
```

---

# 14.9 Stage 7 — M1

Run:

**M0 + Bidirectional LSTM**

Keep the remaining settings fixed.

### Question

> Did bidirectional context help?

Record:

```text
M1 − M0
```

### Commit

```text
add bilstm
```

---

# 14.10 Stage 8 — M2

Run:

**M1 + dropout + recurrent_dropout**

Keep all previous settings fixed.

### Question

> Did regularization improve generalization?

Record:

```text
M2 − M1
M2 − M0
```

Also inspect:

* train/validation gap
* validation curve
* best epoch

### Commit

```text
add dropout
```

---

# 14.11 Stage 9 — M3

Run:

**M2 + GloVe 100d**

### Tasks

* build embedding matrix
* calculate GloVe coverage
* report OOV rate
* verify embedding dimensions
* verify padding/unknown token handling
* keep all other experiment settings unchanged

### Question

> Did pretrained representations improve over random initialization?

Record:

```text
M3 − M2
M3 − M0
```

### Commit

```text
add glove
```

---

# 14.12 Stage 10 — M4

Run:

**M3 + LR scheduling + early stopping + max_len=256**

This is the final recurrent configuration.

### Tasks

* increase maximum context length
* enable learning-rate scheduling
* enable early stopping
* keep checkpointing enabled
* restore best validation weights

### Important

M4 is a bundled optimization/context stage.

Do not claim that the individual contribution of:

* LR scheduling
* early stopping
* max_len

is separately identified.

Only report the combined effect.

### Question

> Does the combined optimization/context stage improve the best recurrent configuration?

Record:

```text
M4 − M3
M4 − M0
```

### Commit

```text
add final lstm
```

---

# 14.13 Stage 11 — Select the Best Recurrent Configuration

Use the predefined primary metric:

**Macro-F1**

Do not select a model because:

* accuracy looks better
* training loss looks better
* it trained faster

unless that is the explicit secondary analysis.

M4 is intended to be the best recurrent configuration, but the results decide whether that is actually true.

### Commit

```text
select best lstm
```

Only make this commit when the selection has been verified.

---

# 14.14 Stage 12 — DistilBERT

Train:

**D0 — DistilBERT fine-tune**

Run:

**3 seeds**

Use the same train/validation/test split.

### Tasks

* tokenize using DistilBERT tokenizer
* fine-tune on the training split
* select using validation Macro-F1
* checkpoint the best model
* evaluate once on the frozen test set
* report mean ± std

### Main comparison

```text
D0 vs M4
```

Secondary comparison:

```text
D0 vs M0
```

### Important

D0 is a model-family benchmark, not another recurrent rung.

### Commit

```text
add distilbert
```

---

# 14.15 Stage 13 — Results Consolidation

All experiment results must come from:

```text
results/runs.csv
```

Do not manually maintain the final comparison table.

### Final table

| Model | Macro-F1 | Accuracy | Δ vs Previous | Δ vs M0 | Interpretation          |
| ----- | -------: | -------: | ------------: | ------: | ----------------------- |
| M0    |          |          |             — |       — | Baseline                |
| M1    |          |          |               |         | Direction               |
| M2    |          |          |               |         | Regularization          |
| M3    |          |          |               |         | Representation          |
| M4    |          |          |               |         | Optimization + context  |
| D0    |          |          |               |         | Model-family comparison |

### Also include

* M0 mean ± std
* D0 mean ± std
* per-class F1
* confusion matrices
* training time
* parameter counts

### Commit

```text
add results table
```

---

# 14.16 Stage 14 — Statistical/Variance Interpretation

Use the three-seed results as a **stability reference**.

Do not present the variance rule as a formal significance test.

When an intermediate delta is small relative to the observed baseline run-to-run spread, describe it cautiously:

> no clear evidence of a meaningful improvement

rather than:

> statistically significant improvement

Unless a formal statistical test has actually been performed.

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

