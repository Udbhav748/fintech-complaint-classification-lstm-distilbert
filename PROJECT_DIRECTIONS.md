# FinTech Complaint Classification: A Controlled LSTM-to-DistilBERT Enhancement Study

## 1. Purpose of This Document

This document is the **single source of truth for the project**.

Any engineer, AI coding assistant, or contributor working on this project must follow the decisions, constraints, architecture, experiment methodology, and coding standards defined here.

Do not introduce major architectural or methodological changes without first checking them against this document.

This project is being developed as an **industry-level AI/ML training project**, not as a simple academic notebook.

The final system must be understandable and defensible by the project owner.

The implementation should therefore prioritize:

* correctness
* reproducibility
* explainability
* clean engineering
* controlled experimentation
* practical resource usage
* simple architecture where possible
* evidence-based decisions

Do not optimize for unnecessary complexity.

---

# 2. Project Title

## Full Title

**FinTech Complaint Classification: A Controlled LSTM-to-DistilBERT Enhancement Study**

## Repository Name

`fintech-complaint-classification-lstm-distilbert`

## Short Project Name

**FinTech Complaint Classifier**

---

# 3. Project Objective

Build a real-world NLP classification system that reads a FinTech customer complaint and predicts its complaint category.

The core project question is:

> **How much can we improve FinTech complaint classification through controlled recurrent-model enhancements, and how does the resulting recurrent model compare with pretrained Transformer transfer learning?**

The project follows the FWC Module 5 Test-6 methodology:

> **Baseline first → record metric → make a controlled enhancement → measure the new metric → calculate the delta → explain the result.**

The FWC material explicitly states that the examiner values the **comparison table and reasoning**, rather than only the final score.

---

# 4. Business Problem

A FinTech company can receive a very large number of customer complaints through text channels.

The operational problem is:

> **Complaint text must be automatically classified so that it can be routed to the correct product/category/team.**

The FWC Module 5 running business case explicitly introduces the problem of large-scale complaint emails needing to reach the correct team and identifies text classification as the NLP solution.

The project therefore treats:

```text
Customer Complaint Text
        ↓
NLP Classifier
        ↓
Complaint/Product Category
```

as the core task.

---

# 5. Project Scope

The project has five major technical stages:

```text
Real CFPB Complaint Data
        ↓
Data Acquisition + Audit
        ↓
Dataset Construction
        ↓
Controlled LSTM Experiments
        ↓
DistilBERT Transfer Learning
        ↓
Comparison + Error Analysis
```

The project is NOT intended to become:

* a chatbot
* a RAG application
* a generative AI application
* an autonomous agent system
* a multi-agent system
* an image model
* a speech model
* an unnecessarily complex MLOps platform

Those technologies are outside the current project scope.

---

# 6. Source Dataset

## Primary Source

Use the:

**Official CFPB Consumer Complaint Database**

Do not use an arbitrary GitHub dataset as the primary project source.

The CFPB database is being used because it provides real-world financial complaint narratives suitable for text classification.

The CFPB source is our **project-selected dataset**.

Important distinction:

> The FWC material establishes the FinTech complaint-classification use case, but it does NOT mandate the CFPB dataset.

Therefore documentation must never falsely state that CFPB was required by the course.

---

# 7. Source Time Range

Use the following source population:

**2023-08-17 → 2026-08-17**

## Why three years?

The three-year window is a project-design choice.

The reasoning is:

* long enough to capture meaningful temporal and linguistic variation
* long enough to reduce dependence on a short-term seasonal spike
* short enough to reduce unnecessary mixing of substantially different complaint/product eras

The three-year window is NOT an FWC requirement.

---

# 8. Initial Product Acquisition Scope

The initial candidate acquisition scope contains these five CFPB Product categories:

1. `Debt collection`
2. `Credit card`
3. `Checking or savings account`
4. `Money transfer, virtual currency, or money service`
5. `Student loan`

These are the **initial acquisition categories**, not automatically the final ML labels.

The FWC material itself gives complaint-classification examples such as billing, loan processing, app issue, and fraud; those examples must not be treated as evidence that CFPB uses the same labels.

---

# 9. Products Initially Excluded

The initial acquisition should exclude:

* Credit reporting or other personal consumer reports
* Credit reporting, credit repair services, or other personal consumer reports
* Mortgage
* Vehicle loan or lease
* Payday loan, title loan, personal loan, or advance loan
* Prepaid card
* Debt or credit management
* Credit card or prepaid card
* Payday loan, title loan, or personal loan
* any other Product outside the five selected categories

---

# 10. Why Credit Reporting Is Excluded Initially

Credit reporting is excluded from the initial scope because its volume is orders of magnitude larger than the other candidate classes.

This matters specifically because **Macro-F1 is a key project metric**.

We do not want an extremely dominant class to distort the experimental problem and weaken the interpretation of minority-class performance.

The FWC material explicitly recommends macro-F1 for imbalanced complaint classification rather than relying only on accuracy.

This is an **initial experimental-scope decision**, not a claim that credit reporting is not a valid FinTech complaint domain.

---

# 11. Label-Set Gate

The five Product categories above are provisional.

The final ML label set MUST NOT be locked before inspecting the actual acquired data.

Before model training, inspect:

* Product
* Sub-product
* Issue
* Sub-issue
* total count per class
* monthly count per class
* class imbalance
* narrative availability
* narrative length
* duplicate behavior
* semantic separation between classes

Then choose between:

### Option A — Native Product Labels

Use selected CFPB Product values directly as ML labels.

### Option B — Smaller Derived Taxonomy

Create a smaller taxonomy only if the native categories are unsuitable for the final modeling problem.

Any mapping must be:

* explicitly documented
* deterministic
* reproducible
* justified by the data and business task

Do not silently merge categories.

---

# 12. Phase Structure

The project is divided into these phases:

```text
Phase 0
Project Definition + Scope
        ↓
Phase 1
Source/API Strategy
        ↓
Phase 2A
CFPB Count Audit
        ↓
Phase 2B
Raw Narrative Acquisition
        ↓
Phase 3
Raw Data Quality Audit + Final Label Set
        ↓
Phase 4
Modeling Dataset Construction
        ↓
Phase 5
LSTM Baseline
        ↓
Phase 6
Controlled LSTM Enhancements
        ↓
Phase 7
DistilBERT Transfer Learning
        ↓
Phase 8
Final Comparison + Error Analysis
        ↓
Phase 9
Documentation + Demo + Viva Preparation
```

No phase should silently perform work belonging to a later phase.

---

# 13. Phase 2A — CFPB Count Audit

## Purpose

Perform a cheap data-volume audit before downloading large amounts of complaint narrative text.

The principle is:

> **Make cheap decisions before expensive work.**

This phase should determine:

* total product volume
* monthly product volume
* obvious sparse periods
* oversized acquisition windows
* whether the initial product scope is feasible

### Phase 2A MUST NOT:

* download narrative text
* preprocess text
* tokenize text
* balance classes
* create training data
* train models
* choose final labels

### Initial engineering thresholds

These are PROJECT HEURISTICS, not CFPB rules.

#### Sparse product-month

Flag:

`< 500 complaints/month`

This is a review flag, not an automatic deletion rule.

#### Oversized product-month

Flag:

`> 80,000 complaints/month`

This means the month should be subdivided into smaller retrieval windows for raw acquisition.

#### Very sparse product overall

Flag:

`< 5,000 complaints across the entire three-year source population`

This becomes a strong candidate for later review during the label-set decision.

Do not automatically delete the class.

---

# 14. Phase 2A API Strategy

Inspect the official CFPB API specification before implementation.

Do not guess:

* endpoint names
* parameter names
* pagination behavior
* response formats
* limits

Determine whether one Trends request can return the monthly breakdown for all selected Products under the shared filters.

If one request can do so, prefer that.

If not, use the minimum number of requests necessary.

The goal is a simple and reproducible count-audit implementation.

---

# 15. Phase 2B — Raw Narrative Acquisition

After Phase 2A:

1. determine practical acquisition windows
2. use the official CFPB source
3. retrieve the narrative-bearing records
4. save the raw files locally
5. preserve the source fields
6. validate coverage
7. stop

## API strategy

For the main acquisition:

* use date-windowed CSV requests
* use non-overlapping date ranges
* avoid dependence on `frm`/`size` pagination beyond verified limits
* save each raw response as an immutable raw file

Monthly windows are preferred when appropriate because they provide a practical balance between:

* payload size
* request count
* reproducibility
* coverage validation

If Phase 2A shows that a month is oversized, split it into smaller windows.

If a month is comfortably below the limit, keep it intact.

---

# 16. Raw Data Must Be Physically Saved

The raw dataset must exist locally.

A trainer should be able to ask:

> "Show me the dataset."

and the project owner should be able to open:

```text
data/raw/cfpb/
```

The project must NOT rely exclusively on a live API request.

The raw source must remain available for inspection and reproducibility.

---

# 17. Required Raw Fields

Preserve all useful CFPB fields where practical.

At minimum retain:

* Complaint ID
* Date received
* Product
* Sub-product
* Issue
* Sub-issue
* Consumer complaint narrative
* Company
* State
* ZIP code, if available
* Submitted via
* Company response to consumer
* Timely response
* Company public response
* Date sent to company

Do not invent unavailable fields.

Do not discard metadata unnecessarily at the raw stage.

---

# 18. Raw / Interim / Processed Separation

Use this structure:

```text
data/
├── raw/
│   └── cfpb/
│       ├── source/
│       ├── manifests/
│       └── README.md
│
├── interim/
└── processed/
```

### RAW

Official CFPB source data.

No destructive cleaning.

### INTERIM

Audited and transformed data used to prepare the modeling dataset.

### PROCESSED

Final train/validation/test datasets ready for experiments.

Never mix these layers.

---

# 19. Raw Acquisition Rules

During raw acquisition, DO NOT:

* lowercase text
* strip punctuation
* remove URLs
* remove stopwords
* lemmatize
* stem
* tokenize
* pad
* truncate
* remove duplicates
* rebalance classes
* apply class weights
* merge labels
* create embeddings
* train models

Raw means raw.

Measure problems rather than silently cleaning them.

---

# 20. Raw Data Manifest

Create:

```text
data/raw/cfpb/manifests/acquisition_manifest.json
```

Record:

* source
* official source URL
* endpoint
* retrieval timestamp
* start date
* end date
* Product filters
* narrative filter
* API query parameters
* retrieval window
* returned record count
* output file
* file size
* checksum/hash if practical
* request status
* failures
* retries
* minimum date
* maximum date
* Product counts

This is the audit trail for the acquisition process.

---

# 21. Raw Data README

Create:

```text
data/raw/cfpb/README.md
```

Document:

* dataset source
* retrieval method
* retrieval date
* source time range
* selected Product categories
* narrative-only filter
* why the CFPB dataset was chosen
* why these products are provisional
* raw file structure
* known limitations

Clearly state:

> The CFPB Consumer Complaint Database is a project-selected real-world dataset aligned with the FinTech complaint-classification use case from the FWC material.

---

# 22. Phase 3 — Raw Data Quality Audit

After acquisition, inspect the real data.

Measure:

### Row-level quality

* total rows
* unique Complaint IDs
* duplicate Complaint IDs
* missing Complaint IDs

### Label quality

* unique Product values
* Product counts
* Sub-product counts
* Issue counts
* unintended categories

### Narrative quality

* missing narratives
* empty narratives
* whitespace-only narratives
* minimum length
* maximum length
* mean length
* median length
* length distribution

### Temporal quality

* minimum Date received
* maximum Date received
* monthly coverage
* missing time windows
* duplicate windows

---

# 23. Final Label Decision

Use the results of Phase 3 to officially lock the label set.

Document:

* chosen labels
* number of samples per label
* why the labels are useful
* why the labels are sufficiently distinct
* why the labels have enough data
* whether any labels were excluded
* whether any mapping was performed

This decision is a **gate**.

No model training should start before this decision is documented.

---

# 24. Dataset Construction Strategy

After the raw audit, create the modeling dataset.

The exact dataset size must be decided from:

* actual available data
* class distribution
* CPU constraints
* training time
* experiment count

Do not arbitrarily force a sample size before inspecting the real data.

The goal is a dataset large enough to produce meaningful experiments but small enough to allow repeated runs.

---

# 25. Compute Constraint

Current development machine:

**CPU-only**

No local NVIDIA CUDA GPU is available.

Therefore the project must be designed for practical CPU experimentation.

### LSTM

Use the larger modeling dataset because LSTM training is comparatively lightweight.

### DistilBERT

Use a **stratified ~8,000–10,000 sample fine-tuning subset** initially.

This is a compute-constrained experimental choice.

Document it honestly.

Do not pretend DistilBERT was fine-tuned on the entire dataset if it was not.

---

# 26. Train / Validation / Test Strategy

Create a reproducible split.

Prefer stratification for the primary experimental split.

Keep the test set isolated from model tuning.

Do not repeatedly optimize against the final test set.

If a temporal validation strategy is added later, document it separately.

The primary requirement is a clean, reproducible evaluation process.

---

# 27. Data Leakage Prevention

Do not allow test data to influence:

* tokenizer construction
* vocabulary decisions
* GloVe coverage decisions
* hyperparameter selection
* class-weight calculations
* model selection

Any data-dependent transformation must be based only on training data where appropriate.

Check for duplicate Complaint IDs across splits.

Where practical, inspect duplicate or highly similar narratives to reduce obvious train/test contamination.

---

# 28. Text Preprocessing

The FWC material presents standard preprocessing concepts including:

* lowercasing
* tokenization
* stopword handling
* stemming/lemmatization
* cleaning URLs/punctuation

It specifically warns that blindly removing negations such as `"not"` can damage NLP tasks where negation changes meaning.

Our pipeline therefore must:

* normalize appropriate text
* remove unnecessary noise
* preserve meaningful negations
* tokenize
* pad/truncate for recurrent models

Do not aggressively preprocess just because it is common NLP practice.

Every preprocessing step must have a reason.

---

# 29. Baseline Experiment — E0

## Model

Simple LSTM.

Architecture:

```text
Complaint Text
    ↓
Tokenizer
    ↓
Padding
    ↓
Random/Trainable Embedding
    ↓
LSTM
    ↓
Dense Classifier
```

This is intentionally simple.

It establishes the baseline.

---

# 30. Baseline Metric

Primary metric:

## Macro-F1

Also record:

* accuracy
* macro precision
* macro recall
* per-class precision
* per-class recall
* per-class F1
* confusion matrix
* validation loss
* training time
* number of parameters

Macro-F1 is the main comparison metric because of the expected class imbalance and the explicit FWC guidance.

---

# 31. Experiment Control Principle

This is one of the most important rules in the project:

> **When testing one enhancement, keep all other relevant variables fixed whenever reasonably possible.**

Do not simultaneously change:

* architecture
* batch size
* optimizer
* learning rate
* sequence length
* embedding strategy
* dropout
* epoch count

unless the experiment is explicitly testing those variables together.

Every experiment must record:

* experiment ID
* model
* change
* configuration
* dataset version
* random seed
* results
* delta
* observation

---

# 32. Delta Calculation

Use:

```text
Δ Macro-F1 =
New Macro-F1 - Previous Macro-F1
```

Also calculate, when useful:

```text
Overall Δ Macro-F1 =
Final Macro-F1 - Baseline Macro-F1
```

Do not report only the final score.

---

# 33. Experiment E1 — Bidirectional LSTM

Change:

```text
LSTM
↓
BiLSTM
```

Hypothesis:

> Bidirectional processing may improve classification because the representation can use information from both directions of the sequence.

The FWC material explicitly demonstrates a Bidirectional wrapper around an LSTM.

Measure:

* Macro-F1
* delta
* per-class impact
* training time
* confusion changes

---

# 34. Experiment E2 — GloVe

Compare:

```text
Random trainable embedding
vs
Pretrained GloVe initialization
```

The FWC material explicitly covers GloVe and pretrained word vectors.

## Important control rule

Prefer:

**trainable random initialization**

versus

**trainable GloVe initialization**

so the primary controlled change is the initialization source.

Do not freeze GloVe unless that is a separate documented experiment.

Record:

* embedding dimension
* vocabulary coverage
* OOV behavior
* trainable/frozen state
* Macro-F1
* delta

---

# 35. Experiment E3 — Class Weights

Use class weights only after inspecting actual class imbalance.

Purpose:

> determine whether minority-class performance improves.

Measure:

* Macro-F1
* minority-class recall
* minority-class F1
* confusion matrix
* accuracy

Do not assume class weights will improve the overall score.

---

# 36. Optional Enhancement Experiments

Test-6 provides additional enhancement options:

* stacked layers
* dropout
* recurrent dropout
* learning-rate scheduling
* early stopping
* longer `max_length`

These are **available experiments**, not mandatory individual rows.

The project should only run them when there is a meaningful hypothesis or observed training behavior that justifies them.

Examples:

### Stacked layers

Test whether additional representational capacity improves performance.

### Dropout / recurrent dropout

Test whether regularization reduces overfitting.

### Learning-rate scheduling

Test whether optimization becomes more stable or effective.

### Early stopping

Test whether training can stop before overfitting.

### Longer `max_length`

Test whether truncation is causing information loss.

---

# 37. Best Enhanced LSTM

After the individual recurrent experiments:

Identify the strongest **defensible enhanced LSTM**.

Do not automatically combine every available technique.

Only keep enhancements supported by measured evidence.

The model should be explainable as:

> "This configuration was selected because these controlled changes produced the strongest validated result under the project constraints."

---

# 38. DistilBERT Experiment

The final major experiment is transfer learning.

The FWC material explicitly demonstrates:

**DistilBERT → complaint classification**

and provides a practical fine-tuning recipe.

Use an appropriate pretrained DistilBERT checkpoint.

Pipeline:

```text
Complaint Text
      ↓
DistilBERT Tokenizer
      ↓
Token IDs + Attention Mask
      ↓
Pretrained DistilBERT
      ↓
Classification Head
      ↓
Fine-tuning
```

---

# 39. DistilBERT CPU Strategy

Because the local environment is CPU-only:

Initial fine-tuning configuration:

* stratified ~8k–10k examples
* 1–2 epochs initially
* small learning rate
* early stopping/evaluation checkpoints
* controlled `max_length`

The FWC material shows a small learning rate and approximately three epochs as a standard recipe; our reduced epoch count is a **compute constraint**, not a change to the theoretical requirement.

If CPU runtime is unacceptable:

1. reduce the DistilBERT subset
2. reduce epochs
3. preserve the experiment methodology
4. document the compute constraint

Never invent results.

---

# 40. DistilBERT Tokenization Rule

The tokenizer must match the selected checkpoint.

The FWC material explicitly warns that mixing tokenizers and checkpoints can corrupt inputs.

Document:

* checkpoint
* tokenizer
* max_length
* padding strategy
* truncation strategy

---

# 41. Max-Length Fairness

Before final model comparison, analyze complaint length.

For the final LSTM and DistilBERT comparison, document:

* chosen max length
* percentage of complaints truncated
* whether both models see comparable usable context

Do not claim a model is better without considering sequence-length differences.

---

# 42. Final Fair Comparison

The headline comparison should be:

```text
Simple LSTM
     vs
Best Enhanced LSTM
     vs
Fine-tuned DistilBERT
```

Do NOT compare:

```text
Untuned LSTM
vs
Highly tuned DistilBERT
```

and call the result a fair architecture comparison.

The goal is to compare:

1. baseline recurrent performance
2. optimized recurrent performance
3. transfer-learning performance

---

# 43. Final Comparison Table

Create one central table:

| Experiment | Change                          | Macro-F1 |     Δ F1 | Accuracy | Training Time | Why / Observation            |
| ---------- | ------------------------------- | -------: | -------: | -------: | ------------: | ---------------------------- |
| E0         | Simple LSTM + random embeddings |   Actual | Baseline |   Actual |        Actual | Baseline                     |
| E1         | BiLSTM                          |   Actual |   Actual |   Actual |        Actual | Actual finding               |
| E2         | + GloVe                         |   Actual |   Actual |   Actual |        Actual | Actual finding               |
| E3         | + Class weights                 |   Actual |   Actual |   Actual |        Actual | Actual finding               |
| E4+        | Optional justified enhancement  |   Actual |   Actual |   Actual |        Actual | Actual finding               |
| Best RNN   | Enhanced LSTM                   |   Actual |   Actual |   Actual |        Actual | Best recurrent configuration |
| Final      | DistilBERT                      |   Actual |   Actual |   Actual |        Actual | Transfer-learning result     |

Do not fabricate numbers.

---

# 44. Reasoning Format for Every Experiment

Every experiment should answer:

### What changed?

### Why did we test it?

### What happened?

### What was the delta?

### Why might the result have happened?

### Did we keep or reject the change?

This is the core viva preparation structure.

---

# 45. Negative Results Are Valid

Never assume that every enhancement must improve performance.

Examples of valid findings:

* GloVe provides little improvement.
* Class weights improve minority recall but reduce overall accuracy.
* Stacked LSTM overfits.
* Longer sequences increase training cost without improving Macro-F1.
* DistilBERT does not outperform the best LSTM on the chosen subset.
* Dropout helps validation performance but reduces training performance.

The project should explain actual results rather than manufacture a positive story.

A surprising but well-explained result is more valuable than an obviously staged improvement.

---

# 46. Evaluation Artifacts

Produce:

## Confusion matrices

For:

* baseline LSTM
* best enhanced LSTM
* DistilBERT

## Learning curves

For LSTM experiments:

* training loss
* validation loss
* training metric
* validation metric

## Per-class metrics

Show:

* precision
* recall
* F1

## Error analysis

Inspect representative misclassified examples.

Look for:

* similar categories
* ambiguous language
* short complaints
* long complaints
* domain-specific wording
* spelling/grammar issues
* multiple issues in one complaint
* rare terminology

---

# 47. Error Analysis Rules

Never cherry-pick only convenient errors.

Select examples systematically.

Document:

* expected label
* predicted label
* complaint text
* likely reason for error

Where raw customer narratives contain sensitive information, avoid unnecessarily displaying personal details.

---

# 48. Experiment Registry

Maintain a machine-readable experiment record.

Suggested file:

```text
reports/experiment_results.csv
```

or:

```text
reports/experiment_results.json
```

Each row/run should include:

* experiment ID
* model name
* architecture
* embedding type
* max_length
* dropout
* recurrent_dropout
* class weights
* optimizer
* learning rate
* scheduler
* epochs
* batch size
* dataset version
* Macro-F1
* precision
* recall
* accuracy
* training time
* parameter count
* notes

The final comparison table should be generated from this record rather than manually typed from memory.

---

# 49. Reproducibility

Every experiment must record:

* random seed
* dataset version
* label mapping version
* preprocessing configuration
* hyperparameters
* model checkpoint
* tokenizer
* runtime environment

A future engineer should be able to understand how a result was produced.

---

# 50. Configuration Management

Avoid hardcoding experiment settings throughout source files.

Prefer:

```text
configs/
├── data.yaml
├── lstm_baseline.yaml
├── lstm_enhanced.yaml
└── distilbert.yaml
```

or a similarly simple configuration structure.

Do not create a configuration framework unless needed.

---

# 51. Recommended Repository Structure

```text
fintech-complaint-classification-lstm-distilbert/
│
├── data/
│   ├── raw/
│   │   └── cfpb/
│   ├── interim/
│   └── processed/
│
├── configs/
│   ├── data.yaml
│   ├── lstm_baseline.yaml
│   ├── lstm_experiments.yaml
│   └── distilbert.yaml
│
├── src/
│   ├── data_acquisition/
│   │   ├── cfpb_counts.py
│   │   └── cfpb_downloader.py
│   │
│   ├── data/
│   │   ├── audit.py
│   │   ├── preprocessing.py
│   │   ├── dataset_builder.py
│   │   └── split.py
│   │
│   ├── models/
│   │   ├── lstm.py
│   │   ├── bilstm.py
│   │   └── distilbert.py
│   │
│   ├── experiments/
│   │   ├── run_lstm.py
│   │   ├── run_bilstm.py
│   │   ├── run_glove.py
│   │   ├── run_class_weights.py
│   │   └── run_distilbert.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       ├── confusion_matrix.py
│       ├── error_analysis.py
│       └── comparison.py
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_baseline_lstm.ipynb
│   ├── 03_lstm_enhancements.ipynb
│   ├── 04_distilbert.ipynb
│   └── 05_final_analysis.ipynb
│
├── reports/
│   ├── experiment_results.csv
│   ├── figures/
│   └── final_report.md
│
├── tests/
│
├── README.md
├── PROJECT_DIRECTIONS.md
├── requirements.txt
└── .gitignore
```

Do not create every file immediately.

Create files when their corresponding phase begins.

---

# 52. Coding Style — Senior Engineer Standard

The code must look like practical production-oriented engineering code, not generated demo code.

## General principles

Prefer:

* clear names
* small functions
* single responsibility
* explicit control flow
* useful logging
* sensible error handling
* type hints where helpful
* deterministic behavior
* reusable components
* minimal abstraction

Avoid:

* unnecessary classes
* deep inheritance
* excessive decorators
* over-engineered frameworks
* giant functions
* duplicated logic
* unexplained constants
* magic numbers
* hidden global state

---

# 53. Naming Style

Use descriptive names.

Good:

```python
complaint_df
product_counts
max_length
class_weights
experiment_results
```

Avoid:

```python
df1
x
tmp
abc
data2
```

Functions should describe actions:

```python
load_complaints()
audit_class_distribution()
build_embedding_matrix()
train_lstm()
evaluate_classifier()
save_experiment_result()
```

---

# 54. Function Design

Prefer small, testable functions.

Bad:

```python
def run_everything():
    # downloads, cleans, trains, evaluates, saves
```

Better:

```python
def load_data():
    ...

def clean_text():
    ...

def build_model():
    ...

def train_model():
    ...

def evaluate_model():
    ...

def save_results():
    ...
```

---

# 55. Error Handling

Never silently ignore exceptions.

Bad:

```python
try:
    ...
except:
    pass
```

Better:

```python
try:
    ...
except requests.RequestException as exc:
    logger.error("CFPB request failed: %s", exc)
    raise
```

Errors must be understandable.

---

# 56. Logging

Use useful logs.

Examples:

```text
Starting CFPB acquisition
Requesting window: 2026-01-01 → 2026-01-31
Product: Credit card
Rows received: 6,812
Saved: data/raw/cfpb/source/...
```

Do not log excessively.

Do not expose unnecessary sensitive narrative text in logs.

---

# 57. Comments

Comments should explain **why**, not restate the code.

Bad:

```python
# increment i
i += 1
```

Good:

```python
# Split oversized monthly windows to stay below the API response threshold.
```

Do not fill every function with obvious comments.

---

# 58. Docstrings

Use concise docstrings for reusable/public functions.

Example:

```python
def audit_product_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return complaint counts grouped by CFPB Product."""
```

Do not write essay-length docstrings for trivial functions.

---

# 59. Type Hints

Use type hints where they improve clarity.

Example:

```python
def load_config(path: Path) -> dict[str, Any]:
    ...
```

Do not add complicated type machinery just to increase code size.

---

# 60. Dependency Discipline

Use the smallest reasonable dependency set.

Preferred core stack:

* Python
* pandas
* numpy
* scikit-learn
* TensorFlow/Keras or PyTorch for LSTM, depending on the chosen implementation
* Hugging Face Transformers
* datasets where useful
* matplotlib
* requests/httpx as appropriate for acquisition

Do not introduce LangChain, LangGraph, agents, vector databases, or unrelated frameworks.

---

# 61. Notebook Style

Notebooks are for:

* exploration
* visualization
* experiment presentation
* result analysis

Reusable logic should live in `src/`.

Do not put the entire project in one notebook.

Notebooks should read like a clear narrative:

```text
1. Objective
2. Load data
3. Audit
4. Prepare data
5. Train
6. Evaluate
7. Compare
8. Conclude
```

---

# 62. Model Code

Model definitions should be separate from training loops.

Example:

```text
src/models/lstm.py
src/models/bilstm.py
src/models/distilbert.py
```

Training should live under experiment scripts.

Do not mix:

* API requests
* data cleaning
* model definition
* training
* evaluation

inside one file.

---

# 63. Test-6 Experimental Ladder

The core experiment sequence should be:

```text
E0
Simple LSTM + random embeddings
        ↓
E1
BiLSTM
        ↓
E2
BiLSTM + GloVe
        ↓
E3
BiLSTM + Class Weights
        ↓
Optional justified enhancements
        ↓
Best Enhanced LSTM
        ↓
DistilBERT
```

This ordering is a project decision.

It is NOT claimed to be an exact order mandated by FWC.

---

# 64. Experiment Hypotheses

Every experiment must have a hypothesis.

Example:

### E1 — BiLSTM

Hypothesis:

> Reading context from both directions may improve classification.

### E2 — GloVe

Hypothesis:

> Pretrained semantic initialization may produce better representations than random initialization.

### E3 — Class Weights

Hypothesis:

> Class weighting may improve minority-class performance.

A hypothesis may be disproven.

That is acceptable.

---

# 65. Model Selection Rule

The best model is NOT necessarily the one with the highest accuracy.

Primary:

**Macro-F1**

Supporting:

* macro precision
* macro recall
* accuracy
* per-class metrics
* confusion matrix
* training cost

The model selected as the best should have a defensible overall trade-off.

---

# 66. Fairness of Comparisons

For every controlled experiment:

* same data split
* same random seed where applicable
* same major hyperparameters unless the experiment changes that parameter
* same evaluation protocol
* same target labels

When a difference cannot be held fixed, document it explicitly.

---

# 67. Training Time

Record training time for meaningful experiments.

This matters because this is an industry-oriented project.

Do not evaluate only quality.

Consider:

* performance
* training cost
* parameter count
* inference considerations

---

# 68. DistilBERT Fallback Strategy

Because development is CPU-only:

### First attempt

~8k–10k stratified training subset.

### Initial epochs

1–2.

### If still too slow

Reduce:

1. dataset size
2. sequence length if justified
3. epoch count

Do not fabricate or extrapolate results.

Document the compute limitation.

---

# 69. GloVe Fallback Strategy

Use a practical pretrained GloVe configuration.

If downloading/processing the preferred GloVe package becomes a blocking issue:

1. use a smaller standard GloVe configuration
2. document the embedding dimension
3. measure the actual result
4. continue

If GloVe genuinely cannot be completed within the time constraint:

* complete the remaining experiments
* document GloVe as pending
* do not invent its metric

---

# 70. Time-Budget Principle

The project has a short implementation deadline.

Therefore:

> **Prefer a complete, defensible experiment pipeline over a partially implemented "perfect" architecture.**

Do not spend most of the available time building sophisticated infrastructure that does not improve the Test-6 deliverable.

Priority order:

1. real data
2. correct labels
3. baseline
4. controlled enhancements
5. DistilBERT
6. comparison
7. documentation/demo

---

# 71. What the AI Coding Assistant Is Allowed to Do

The assistant may:

* write implementation code
* create scripts
* create utilities
* add tests
* debug errors
* improve code quality
* create plots
* generate documentation drafts
* run commands
* inspect outputs

But the assistant must NOT silently decide major project policy.

It should ask or stop when encountering:

* label-set decisions
* major dataset-scope changes
* methodology changes
* metric changes
* experiment-order changes
* data leakage risks
* destructive data operations

---

# 72. Human Decision Rule

Every major decision must be understandable by the project owner.

The project owner must be able to explain:

* why CFPB
* why three years
* why the initial product pool
* why credit reporting is excluded initially
* why count audit comes first
* why monthly acquisition windows are used
* why Macro-F1
* why LSTM baseline
* why BiLSTM
* why GloVe
* why class weights
* why DistilBERT
* why the final model was selected

Do not use a technique merely because an AI assistant suggested it.

---

# 73. AI-Assisted Development Rule

AI coding tools may assist with implementation, but:

> **No code should be accepted without understanding what it does.**

For every major generated component:

1. inspect the code
2. understand the inputs
3. understand the outputs
4. understand the important decisions
5. run it
6. validate the result
7. then commit it

The goal is a project the owner can defend naturally.

---

# 74. Trainer/Viva Readiness

Every major project component should have a simple answer to:

### What is it?

### Why did we use it?

### What problem does it solve?

### What did we compare it against?

### What changed?

### What was the metric delta?

### What did we learn?

This applies to both data engineering and ML.

---

# 75. Git Strategy

Use meaningful commits.

Example:

```text
feat: add CFPB count audit
feat: add CFPB raw acquisition pipeline
feat: add dataset audit
feat: add baseline LSTM
feat: add BiLSTM experiment
feat: add GloVe experiment
feat: add class-weight experiment
feat: add DistilBERT fine-tuning
feat: add experiment comparison report
docs: add project methodology
```

Avoid meaningless commits:

```text
update
changes
final
test
done
```

---

# 76. Git Safety

Never commit:

* raw CFPB datasets
* large local datasets
* model caches
* virtual environments
* secrets
* API keys
* temporary files

Use `.gitignore`.

Keep source data locally.

---

# 77. Documentation Requirements

The final README should contain:

1. Project overview
2. Business problem
3. Dataset/source
4. Data acquisition method
5. Final label set
6. Preprocessing
7. Baseline
8. Enhancement experiments
9. DistilBERT
10. Evaluation
11. Comparison table
12. Error analysis
13. Limitations
14. How to reproduce
15. Project structure

---

# 78. Final Deliverables

The completed project should contain:

### Data

* raw CFPB source data
* data acquisition manifest
* data audit report
* final modeling dataset

### Models

* baseline LSTM
* enhanced recurrent model(s)
* best enhanced LSTM
* fine-tuned DistilBERT

### Evaluation

* Macro-F1
* precision
* recall
* accuracy
* confusion matrices
* learning curves
* experiment comparison table
* delta analysis
* error analysis

### Documentation

* README
* PROJECT_DIRECTIONS.md
* experiment documentation
* data documentation
* reproducibility information

---

# 79. Final Project Story

The completed project must tell one coherent story:

```text
Real-world CFPB complaint data
        ↓
Count audit
        ↓
Data-quality audit
        ↓
Final label decision
        ↓
Modeling dataset
        ↓
Simple LSTM baseline
        ↓
Measure Macro-F1
        ↓
BiLSTM
        ↓
Measure Δ
        ↓
GloVe
        ↓
Measure Δ
        ↓
Class weights
        ↓
Measure Δ
        ↓
Best Enhanced LSTM
        ↓
DistilBERT transfer learning
        ↓
Final comparison
        ↓
Error analysis
        ↓
Business/engineering conclusion
```

The project is about **measuring the effect of model improvements**, not simply finding one model that gives the highest number.

---

# 80. Definition of Success

The project is successful when a trainer can ask:

> "What problem are you solving?"

and we can explain the business problem.

> "Where did your data come from?"

and we can show the actual CFPB source data.

> "Why did you choose these labels?"

and we can explain the data-driven label decision.

> "What was your baseline?"

and we can show the simple LSTM result.

> "How did you improve it?"

and we can show controlled experiments.

> "What was the improvement?"

and we can show the Macro-F1 deltas.

> "Why did it improve?"

and we can explain the mechanism.

> "Why did you use DistilBERT?"

and we can explain transfer learning.

> "Did DistilBERT definitely win?"

and we can honestly explain the actual result.

> "Can you reproduce it?"

and we can show the configuration, dataset version, experiment registry, and code.

---

# 81. Non-Negotiable Rules

1. Do not fabricate metrics.
2. Do not fabricate dataset counts.
3. Do not fabricate API behavior.
4. Do not claim a course requirement that the FWC material does not state.
5. Do not silently change the label set.
6. Do not leak test data.
7. Do not compare unfairly tuned models.
8. Do not hide negative experiments.
9. Do not commit raw datasets to public GitHub.
10. Do not use unnecessary architecture or frameworks.
11. Do not make major methodological decisions without documentation.
12. Do not move to the next phase until the current phase has produced its required artifact.

---

# 82. Current Execution State

```text
Phase 0 — Project Scope             LOCKED
Phase 1 — Source/API Strategy      LOCKED
Phase 2A — Count Audit             COMPLETE
Phase 2B — Raw Acquisition         COMPLETE
Phase 3 — Data Audit/Labels        NEXT
Phase 4 — Dataset Construction     PENDING
Phase 5 — LSTM Baseline            PENDING
Phase 6 — Enhancements             PENDING
Phase 7 — DistilBERT               PENDING
Phase 8 — Final Comparison         PENDING
Phase 9 — Documentation/Demo       PENDING
```

Do not train models before the final label set is documented.

## Phase 2A findings carried forward

Verified against the live API on 2026-08-18:

1. **A single `/trends` request covers the whole audit.** `lens=product` requires
   a `focus` or `sub_lens`, so `sub_lens=sub_product` is supplied; counts are
   read from `aggregations.product.product.buckets[].trend_period.buckets[]`.
2. **`has_narrative=yes` is wrong.** The published swagger spec documents
   `yes`/`no`, but those values return HTTP 424 from the live endpoint. The
   working value is `true`.
3. **A custom `User-Agent` header triggers the CDN WAF** (HTTP 403). The default
   client User-Agent must be left in place.
4. **552,157 narrative-bearing complaints** across the five products; no product
   is below the very-sparse threshold, so the provisional scope is feasible.
5. **2026-07 is incomplete** because of the CFPB publication lag. The effective
   end of the source population is earlier than the nominal 2026-08-17, and
   Phase 2B must record the effective end date it actually uses.
6. **Money transfer 2025-01 is a 44x volume spike** (41,618 against a median of
   944). This requires a sampling decision in Phase 4 so one month does not
   dominate that class and distort macro-F1.

## Phase 2B findings carried forward

Retrieved 2026-08-18. The scripted per-month downloader was built and tested
end to end (see `src/data_acquisition/cfpb_downloader.py`), but the CFPB edge
(Akamai) began returning HTTP 403 to this client across the whole domain
before the real acquisition could run. Rather than attempt to work around the
block, the same five product windows were retrieved manually via the official
search CSV export (one request per product, covering its full planned range)
and validated with `--verify`, which runs the identical integrity checks and
compares coverage against the Phase 2A counts by row content rather than by
request shape.

1. **107,992 rows acquired**, matching the plan exactly. All 58 planned
   product-months match their Phase 2A expected count with zero delta.
2. **No duplicate Complaint IDs, no missing narratives.** 107,992 unique rows.
3. **`date_received_max` is inclusive.** Confirmed empirically: the final
   month of every product's window retrieved its full expected count.
4. **`format=csv` was not capped by `size=60000`** at these window sizes
   (largest single file: 24,007 rows for Debt collection).
5. Acquisition manifest with per-file checksums:
   `data/raw/cfpb/manifests/acquisition_manifest.json`. Full report:
   `reports/phase2b_acquisition_report.md`.

---

# 83. Current Immediate Task

## Phase 3 — Raw Data Quality Audit + Final Label Set

Phase 2B is complete. Using the acquired raw extract under
`data/raw/cfpb/source/`, run the Phase 3 audit exactly as specified in
sections 22-23:

1. row-level quality (duplicate/missing Complaint IDs)
2. label quality (Product, Sub-product, Issue, Sub-issue counts)
3. narrative quality (missing/empty/whitespace-only, length distribution)
4. temporal quality (coverage, gaps)

Then lock the final label set (Option A native labels vs Option B derived
taxonomy) and document the decision per section 23. Do not start Phase 4
(dataset construction) before this decision is documented.

---

# 84. Final Engineering Principle

The project should always follow this cycle:

```text
Understand
    ↓
Decide
    ↓
Implement
    ↓
Run
    ↓
Inspect
    ↓
Explain
    ↓
Validate
    ↓
Proceed
```

The objective is not to create the most complicated AI system.

The objective is to create a **real, reproducible, industry-style NLP project whose technical decisions can be clearly explained and defended by its owner.**
