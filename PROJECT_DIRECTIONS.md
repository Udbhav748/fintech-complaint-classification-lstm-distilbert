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

# 25A. Compute Environment Strategy

> **Compute Environment Strategy:** Recurrent experiments are executed locally
> on CPU for consistency and practical iteration. DistilBERT fine-tuning may be
> executed on a GPU-enabled cloud environment such as Kaggle when required by
> compute constraints. The dataset version, label mapping, tokenizer
> configuration, evaluation protocol, and experiment configuration must remain
> identical across environments. **Environment changes are considered
> infrastructure differences, not methodological changes.**

## Division of work

```text
                PROJECT
                   |
       +-----------+-----------+
       |                       |
 Local CPU                  Kaggle GPU
       |                       |
 E0 Simple LSTM             E7 DistilBERT
 E1 BiLSTM
 E2 Stacked BiLSTM
 E3 GloVe
 E4 Regularization
 E5 Sequence length
 E6 Class weights
       |                       |
       +-----------+-----------+
                   |
          Same evaluation pipeline
                   |
          Macro-F1 + delta + reasoning
```

## What must remain identical across environments

* dataset version (`cfpb_phase4_v1`)
* train / validation / test split and the seed that produced it
* label mapping
* tokenizer configuration
* evaluation protocol and metric definitions
* experiment registry schema

## Viva answer this supports

> *"Why did you train the LSTM family locally but DistilBERT on Kaggle?"*

> "The recurrent models were computationally manageable on the local CPU, while
> DistilBERT fine-tuning was moved to a GPU environment to avoid unnecessary CPU
> bottlenecks. The data, configuration, evaluation protocol, and experiment
> definition remained controlled."

## Local development machine notes (not methodology)

Machine management steps — stopping unused containers to free RAM, clearing
caches, relocating cold data — are **development-environment optimizations**.
They are deliberately **not** recorded as experiments and must not appear in the
experiment registry or the comparison table.

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

## Verified E0 result (COMPLETE — do not modify retroactively)

Recorded in `reports/experiment_results.csv` and
`reports/phase5_e0_baseline_report.md`:

| Metric | Verified value |
| --- | ---: |
| **Macro-F1 (primary)** | **0.8435** |
| Accuracy | **84.45%** |
| Macro Precision | **0.8440** |
| Macro Recall | **0.8438** |
| Training time | 4576.32 s |
| Total / trainable parameters | 3,249,989 |

Baseline configuration: `cfpb_phase4_v1`, vocabulary 25,000, embedding dim 128,
LSTM units 64, `max_length` 128, batch size 64, learning rate 0.001, 10 epochs,
seed 42, random trainable embeddings, no dropout, no class weights.

Baseline architecture:

```text
Complaint text
    ↓
training-only vocabulary
    ↓
padding / truncation
    ↓
random trainable embedding
    ↓
single-layer unidirectional LSTM
    ↓
5-class classifier
```

> **All E1+ experiments must reference E0.**
> **Do not modify E0 retroactively.** If the baseline is ever re-run under
> changed conditions, it becomes a new experiment ID, not a revised E0.

Observed E0 training behavior carried forward as evidence for E4: validation
loss minimum at epoch 4 (0.4175), validation Macro-F1 peak at epoch 6 (0.8513),
divergence thereafter.

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

# 32A. Core Experimental Philosophy

This project is **not** simply a sequence of increasingly complex models.

Its purpose is to demonstrate a disciplined NLP experimentation process:

```text
Baseline
    ↓
identify weaknesses
    ↓
form a hypothesis
    ↓
change one meaningful factor
    ↓
measure the new result
    ↓
calculate the metric delta
    ↓
analyze per-class effects
    ↓
inspect computational cost
    ↓
explain why the result changed
    ↓
decide whether to retain or reject the change
    ↓
continue to the next experiment
```

The central Test-6 rule remains unchanged:

> **Baseline first → record metric → enhance → show delta.**

The FWC material explicitly states that the examiner values the **comparison
table and the reasoning**, not simply the final score.

## How the enhancement menu is interpreted

The enhancement list in the FWC material is a **menu of valid enhancement
techniques**. It is *not* interpreted as a requirement that every technique
must become a separate experiment.

However, because this is a graded assessment and the goal is to demonstrate
strong engineering effort, this project deliberately performs a **broader and
more visible experimental study than the strict minimum**.

---

# 32B. Full Experimental Strategy

The intended experimental ladder is:

```text
E0 — Simple LSTM Baseline
E1 — Bidirectional LSTM
E2 — Stacked Bidirectional LSTM
E3 — GloVe Embedding Experiment
E4 — Dropout / Recurrent Dropout / Regularization Experiment
E5 — Sequence-Length Experiment
E6 — Class-Weight Experiment
E7 — DistilBERT Fine-Tuning
```

**The exact implementation of E4 and E5 must be evidence-driven.**

Do not force an enhancement simply to create another row in the table.

* dropout / recurrent_dropout must be justified by observed overfitting or
  generalization behavior
* early stopping may be folded into the regularization/training study
* learning-rate scheduling may be tested only if optimization behavior
  justifies it
* longer `max_length` must be evaluated against the actual
  tokenizer/sequence-length analysis
* class weights must be tested honestly **even though the expected delta is
  small**, because the acquired dataset is nearly balanced

## Three categories in the final presentation

The final write-up must clearly distinguish:

1. **required / core experiments** — E0, E1, E7 and the enhancements that
   directly answer the Test-6 requirement
2. **evidence-driven supporting experiments** — enhancements run because the
   measured behavior of an earlier experiment justified them
3. **optional experiments considered but not run** — with the reason they were
   judged unnecessary

Category 3 is not a gap. Documenting a technique that was considered and
rationally declined is itself evidence of engineering judgement.

---

# 33. Experiment E1 — Bidirectional LSTM

**Question:** Does bidirectional context improve classification over the
unidirectional LSTM?

Controlled change:

```text
LSTM
↓
BiLSTM
```

All other relevant settings remain fixed.

Hypothesis:

> Bidirectional processing may improve classification because the
> representation can use information from both directions of the sequence.

The FWC material explicitly demonstrates a Bidirectional wrapper around an LSTM.

Analyze:

* Macro-F1
* Δ Macro-F1 against E0
* accuracy
* macro precision
* macro recall
* per-class F1
* confusion matrix
* training time
* parameter count
* inference cost

**Pay particular attention to the `Checking or savings account` ↔
`Money transfer, virtual currency, or money service` confusion observed in
E0.** If bidirectional context helps anywhere, it should help there.

Status: **PENDING — TO BE MEASURED**

---

# 33A. Experiment E2 — Stacked Bidirectional LSTM

**Question:** Does additional recurrent depth improve representation quality
beyond a single BiLSTM?

Conceptual structure:

```text
Embedding
    ↓
BiLSTM
    ↓
second recurrent layer
    ↓
classifier
```

Do not change unrelated hyperparameters.

Analyze whether the additional capacity:

* improves Macro-F1
* causes overfitting
* increases training time
* increases parameter count
* improves the difficult classes specifically
* produces diminishing returns

**If the stacked model performs worse, keep the negative result and explain
it.** Depth that does not pay for itself is a legitimate and reportable
finding.

Status: **PENDING — TO BE MEASURED**

---

# 34. Experiment E3 — GloVe Embedding Experiment

**Question:** Does pretrained GloVe initialization improve classification
relative to random initialization?

Controlled comparison:

```text
Random trainable embeddings
vs
Trainable GloVe-initialized embeddings
```

The primary comparison must isolate **initialization source** as far as
reasonably possible.

The FWC material explicitly covers GloVe and pretrained word vectors.

## Important control rule

Prefer:

**trainable random initialization**

versus

**trainable GloVe initialization**

so the primary controlled change is the initialization source.

**Do not freeze GloVe unless freezing is intentionally introduced as a
separate documented experiment.**

## Existing Phase 4 evidence (verified — do not restate generically)

Measured in `reports/phase4_modeling_dataset_report.md` against the training
vocabulary, using **GloVe 6B 100d**:

| Measure | Verified value |
| --- | ---: |
| Unique word coverage | **81.11%** (20,277 matched words) |
| Token-frequency coverage | **99.11%** |

Interpretation to carry into the hypothesis: although roughly one in five
*unique* vocabulary words has no GloVe vector, those misses are overwhelmingly
rare words — **99.11% of actual token instances are covered**. The unmatched
tail is therefore low-frequency, which tempers how much gain should be expected
from GloVe initialization.

Record:

* GloVe dimension
* vocabulary coverage
* token-frequency coverage
* OOV rate
* trainable/frozen state
* Macro-F1
* Δ Macro-F1
* per-class effects
* training behavior

Status: **PENDING — TO BE MEASURED**

---

# 34A. Experiment E4 — Regularization / Generalization Study

**Question:** Does regularization improve generalization, rather than merely
reducing training accuracy?

## Evidence that motivates this experiment (verified)

This experiment is **not speculative**. The E0 baseline learning curves in
`reports/phase5_e0_baseline_report.md` already record mild overfitting:

* validation loss reached its global minimum at **Epoch 4 (0.4175)**
* validation Macro-F1 peaked at **Epoch 6 (0.8513)**
* beyond Epoch 6 training loss kept falling (0.2924 to 0.1610, train accuracy
  94.65%) while validation loss rose (0.4221 to 0.5361)
* the recorded E0 test Macro-F1 is **0.8435**

The gap between the epoch-6 validation peak and the final recorded score is the
concrete, measured symptom this experiment exists to address.

Potential components, tested deliberately rather than randomly:

```text
Baseline recurrent architecture
    ↓
dropout
    ↓
dropout + recurrent_dropout
    ↓
early stopping if appropriate
```

Measure:

* training loss
* validation loss
* the training/validation metric gap
* Macro-F1
* Δ Macro-F1
* **epoch at best validation performance**
* training time

**If one configuration is sufficient to answer the question, do not create
unnecessary additional rows.**

Status: **PENDING — TO BE MEASURED**

---

# 34B. Experiment E5 — Sequence-Length Study

**Question:** Does preserving more sequence context improve complaint
classification enough to justify the additional computational cost?

## Existing Phase 4 evidence (verified — use these, do not invent)

Measured against real tokenizer output in
`reports/phase4_modeling_dataset_report.md`. Sub-word tokenization produces
approximately **1.28x** more tokens than whitespace words.

| `max_length` | % Truncated | % Retained Intact | Compute note |
| ---: | ---: | ---: | --- |
| 64 | 90.18% | 9.82% | Discards detail in >90% of complaints |
| **128** | **72.45%** | **27.55%** | Current baseline for E0/E1 |
| 256 | 41.46% | 58.54% | ~4x attention cost of L=128, O(L^2) |
| 512 | 11.80% | 88.20% | Prohibitive for CPU fine-tuning |

Candidate comparison:

```text
max_length = 128
vs
max_length = 256
```

Optionally include another justified value if the evidence supports it.

For every candidate, record:

* percentage of sequences truncated
* Macro-F1
* Δ Macro-F1
* training time
* memory / compute implication

The final interpretation **must discuss the quality/compute trade-off**.

> **Do NOT select a longer sequence purely because it produces a larger
> number.** A gain that costs 4x compute must be argued, not assumed.

Status: **PENDING — TO BE MEASURED**

---

# 35. Experiment E6 — Class Weights

**Question:** Does class weighting improve minority-class performance?

Use **training-only** class weights. Do not calculate weights from validation
or test data.

## Important project fact (verified in Phase 3)

The Phase 2B acquisition intentionally produced an approximately balanced
modeling population: **1.19x imbalance in this extract, against 8.3x in the
real CFPB population.** The class-weight values are therefore close to 1.

**Expected result: a small or negligible delta.**

This is **NOT** a failed experiment. A near-zero effect must be documented
honestly and explained as a direct consequence of the acquired class
distribution.

> **Do not rebalance the dataset simply to force class weights to produce an
> improvement.** Doing so would invalidate the comparison against every other
> experiment in the ladder.

Analyze:

* Macro-F1
* Δ Macro-F1
* macro recall
* per-class recall
* per-class F1
* confusion matrix

Status: **PENDING — TO BE MEASURED**

---

# 36. Optional Enhancement Experiments (Considered, Not Automatically Run)

Test-6 provides a menu of enhancement options. Several have been **promoted
into the numbered ladder** because there is measured evidence justifying them:

| Technique | Disposition |
| --- | --- |
| stacked layers | promoted to **E2** |
| dropout / recurrent dropout | promoted to **E4** |
| early stopping | folded into **E4** |
| longer `max_length` | promoted to **E5** |
| class weights | promoted to **E6** |
| learning-rate scheduling | **remains optional** |

The remaining optional technique should only be run when there is a meaningful
hypothesis or observed training behavior that justifies it:

### Learning-rate scheduling

Test whether optimization becomes more stable or effective. Run this only if
the E0-E4 loss curves show instability, oscillation, or a plateau that a
schedule would plausibly address. If it is not run, **record it in the
"considered but not run" category with the reason.**

---

# 37. Best Enhanced LSTM

After the individual recurrent experiments:

Identify the strongest **defensible enhanced LSTM**.

Do not automatically combine every available technique.

Only keep enhancements supported by measured evidence.

The model should be explainable as:

> "This configuration was selected because these controlled changes produced the strongest validated result under the project constraints."

---

# 38. Experiment E7 — DistilBERT Transfer Learning

This is the transfer-learning stage and the final experiment in the ladder.

Execution environment: **local CPU strategy remains documented as the fallback
(sections 25, 25A, 39), with Kaggle GPU as the preferred environment.**

The subset size actually used, and the environment it ran in, must be recorded
with the result. Do not assume the CPU-era subset size still applies if the
experiment is run on GPU — record what was actually done.

Record for E7:

* pretrained checkpoint
* tokenizer
* `max_length`
* learning rate
* epochs
* batch size
* training subset size **and the environment it ran in**
* Macro-F1, precision, recall, accuracy
* training time
* inference time where practical

> **Data asymmetry must be stated explicitly.** If DistilBERT is fine-tuned on
> a smaller stratified subset while the LSTM family uses the larger modeling
> dataset, this MUST be stated in the final report and MUST NOT be presented as
> an equal-data training comparison.

Status: **PENDING — TO BE MEASURED**


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

| Experiment | Change | Macro-F1 | Δ Macro-F1 | Accuracy | Precision | Recall | Training Time | Parameters | Main Finding |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E0 | Simple LSTM | **0.8435** | — | **84.45%** | **0.8440** | **0.8438** | **4576.32 s** | **3,249,989** | Reference baseline |
| E1 | BiLSTM | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| E2 | Stacked BiLSTM | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| E3 | GloVe | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| E4 | Regularization | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| E5 | Longer max_length | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| E6 | Class weights | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| E7 | DistilBERT | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Best RNN | Best enhanced recurrent model | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | Selected configuration |

The E0 row is populated from the verified `reports/experiment_results.csv` row
and `reports/phase5_e0_baseline_report.md`.

**All other values must come from actual runs. Never fabricate results.** Until
an experiment has been executed, its cells stay `PENDING`.

Where an experiment runs in a different compute environment (section 25A), note
the environment alongside the training time so the comparison is not misread as
equal-hardware.

---

# 43A. Decision-Driven Experimentation

Every major experiment must be documented with this exact structure. **This
structure is mandatory.**

### 1. Objective
What are we testing?

### 2. Hypothesis
Why do we expect this change to help?

### 3. What changed?
Exactly one major intended change wherever possible.

### 4. What stayed fixed?
List explicitly: dataset, split, tokenizer/vocabulary, preprocessing, seed,
optimizer, learning rate, batch size, epochs, `max_length`, embedding strategy,
and any other relevant control.

### 5. Result
Actual metrics.

### 6. Delta
New Macro-F1 minus previous Macro-F1.

### 7. Per-class impact
Which classes improved or worsened?

### 8. Error/confusion impact
Did the dominant error modes change?

### 9. Computational impact
Training time, parameter count, inference cost where relevant.

### 10. Interpretation
Why did the result likely happen?

### 11. Decision
**KEEP / REJECT / INCONCLUSIVE.**

---

# 43B. Controlled Variables Table

Every experiment write-up must make it **obvious which single variable
changed**. Maintain this table across the ladder:

| Variable | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dataset | fixed | fixed | fixed | fixed | fixed | fixed | fixed | subset — state size |
| Split | fixed | fixed | fixed | fixed | fixed | fixed | fixed | fixed |
| Seed | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 |
| Vocabulary | same | same | same | same | same | same | same | DistilBERT tokenizer |
| Preprocessing | same | same | same | same | same | same | same | same |
| Optimizer | same | same | same | same | same | same | same | state |
| Learning rate | same | same | same | same | same | same | same | state |
| Batch size | same | same | same | same | same | same | same | state |
| Epochs | same | same | same | same | varies | same | same | state |
| `max_length` | 128 | 128 | 128 | 128 | 128 | **varies** | 128 | 128 initially |
| Compute env | local CPU | local CPU | local CPU | local CPU | local CPU | local CPU | local CPU | **GPU** |
| Intended change | baseline | BiLSTM | stacked depth | GloVe init | regularization | sequence length | class weights | transfer learning |

Cells that legitimately differ (E5 `max_length`, E7 tokenizer and environment)
are the **intended** change for that row and must be called out as such.

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

**An experiment does not need to improve Macro-F1 to be valuable.**

Never assume that every enhancement must improve performance.

Examples of valid findings:

* BiLSTM performs worse than the unidirectional LSTM.
* Stacked layers overfit and reduce generalization.
* GloVe provides little or no improvement.
* Class weights have a near-zero effect.
* Longer `max_length` increases compute without improving Macro-F1.
* DistilBERT does not outperform the best LSTM under the constrained setup.
* Dropout helps validation performance but reduces training performance.

## Required handling when an experiment does not improve the metric

1. **keep the experiment** — do not delete it from the ladder or the table
2. **record the actual metric**
3. **calculate the delta** even when it is negative
4. **inspect where performance changed** (per-class, confusion matrix)
5. **explain why the result may have occurred**
6. **decide whether the technique should be retained** — KEEP / REJECT /
   INCONCLUSIVE

> **Never manipulate an experiment to force a positive result.**

The project should explain actual results rather than manufacture a positive story.

A surprising but well-explained result is more valuable than an obviously staged improvement.

---

# 45A. Evidence Before Decision

Every major project decision must be supported by **visible evidence** wherever
possible.

| Decision | Must be supported by |
| --- | --- |
| Label set | Product / Sub-product / Issue / Sub-issue analysis |
| `max_length` | actual tokenizer truncation analysis |
| Regularization | learning curves and validation behavior |
| Class weights | class distribution and training-only weights |
| Architecture | per-class metrics, confusion matrix, Macro-F1 delta |
| Model selection | quality + compute + error behavior together |

The project must **not** say:

> "we chose X because it is industry standard."

It must say:

> "we tested/observed X under these conditions and chose it because..."

---

# 45B. Effort and Engineering Work Requirements

The project must visibly demonstrate that substantial analysis and engineering
effort went into the solution.

### Data investigation

CFPB source audit; Product distribution analysis; Sub-product analysis; Issue
analysis; Sub-issue analysis; narrative length analysis; temporal analysis;
duplicate analysis; conflicting-label analysis.

### Data engineering

Reproducible acquisition process; raw/interim/processed separation; manifest;
dataset versioning; leakage-safe grouping; deterministic split; tokenizer
analysis.

### Model experimentation

Baseline architecture; architecture ablation; embedding ablation;
regularization study; sequence-length study; class-weight study;
transfer-learning study.

### Model evaluation

Macro-F1; accuracy; precision; recall; per-class metrics; confusion matrices;
learning curves; error analysis; training cost.

### Documentation

Experiment registry; experiment reports; notebooks; final comparison table;
final methodology explanation; limitations; viva-ready reasoning.

> **Do not add technology merely to increase the apparent size of the project.**
> The effort must be visible through actual analysis and controlled
> experimentation.

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

# 46A. Targeted Error Analysis Requirements

Targeted error analysis is required after each major model stage.

At minimum, compare:

* **E0 baseline**
* **Best Enhanced LSTM**
* **E7 DistilBERT**

Analyze for each:

* most confused class pairs
* representative false positives
* representative false negatives
* long-text errors
* short-text errors
* ambiguous complaints
* domain-specific language
* duplicate / boilerplate behavior

Use **actual predictions**.

> **Do not cherry-pick only favorable examples.** A representative error sample
> includes the cases the model got embarrassingly wrong.

Known starting point from E0: the `Checking or savings account` and
`Money transfer, virtual currency, or money service` pair is the dominant
confusion to track across the ladder.

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
Bidirectional LSTM
        ↓
E2
Stacked Bidirectional LSTM
        ↓
E3
GloVe embedding experiment
        ↓
E4
Regularization / generalization study
        ↓
E5
Sequence-length study
        ↓
E6
Class-weight experiment
        ↓
Best Enhanced LSTM
        ↓
E7
DistilBERT transfer learning
```

This ordering is a project decision.

It is NOT claimed to be an exact order mandated by FWC.

---

# 64. Experiment Hypotheses

Every experiment must have a hypothesis.

Example:

### E1 — BiLSTM

> Reading context from both directions may improve classification, particularly
> for the Checking/Savings vs Money transfer confusion seen in E0.

### E2 — Stacked BiLSTM

> Additional recurrent depth may improve representation quality — or may
> overfit, given E0 already shows mild overfitting.

### E3 — GloVe

> Pretrained semantic initialization may produce better representations than
> random initialization. Tempered expectation: token-frequency coverage is
> already 99.11%, so the uncovered tail is rare words.

### E4 — Regularization

> Dropout / recurrent dropout / early stopping may close the gap between the
> epoch-6 validation peak (0.8513) and the final recorded E0 score (0.8435).

### E5 — Sequence Length

> Raising `max_length` from 128 to 256 cuts truncation from 72.45% to 41.46%
> and may improve Macro-F1 — at roughly 4x attention cost.

### E6 — Class Weights

> Class weighting may improve minority-class performance. Expected delta is
> near zero because this extract is only 1.19x imbalanced by acquisition design.

### E7 — DistilBERT

> Pretrained transformer representations may outperform the best recurrent
> model, subject to the stated data asymmetry.

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
Real FinTech complaint data
        ↓
data investigation
        ↓
label decision
        ↓
leakage-safe dataset construction
        ↓
simple LSTM baseline (E0)
        ↓
identify weaknesses
        ↓
architecture enhancement (E1 BiLSTM, E2 stacked)
        ↓
representation enhancement (E3 GloVe)
        ↓
regularization / generalization experiments (E4)
        ↓
sequence-length experiment (E5)
        ↓
class-weight experiment (E6)
        ↓
best recurrent model
        ↓
DistilBERT transfer learning (E7)
        ↓
comparison
        ↓
error analysis
        ↓
final engineering conclusion
```

Every arrow between experiments carries a **measured delta and a written
reason**, not merely a new model.

The project is therefore simultaneously:

1. **an NLP classification system**
2. **a controlled model-enhancement study**

The project is about **measuring the effect of model improvements**, not simply finding one model that gives the highest number.

---

# 79A. Viva-Defensibility Requirement

The project owner must be able to answer every question below, **with evidence
in the documentation** rather than from memory or general knowledge:

### Data and scope
* Why CFPB?
* Why three years?
* Why these Products?
* Why Product instead of Issue?
* Why group-aware splitting?
* Why Macro-F1?

### Modeling ladder
* Why an LSTM baseline?
* Why BiLSTM?
* Why stacked layers?
* Why GloVe?
* Why regularization?
* Why class weights?
* Why longer `max_length`?
* Why DistilBERT?

### Comparison integrity
* Why is the DistilBERT training set smaller?
* Why was DistilBERT trained in a different compute environment?
* Which enhancement helped most?
* Which enhancement failed?
* Why?
* What is the main remaining error?
* What is the compute trade-off?

For each question the documentation must contain the **supporting evidence** —
a table, a figure, a metric delta, or a recorded decision — not an assertion.

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
Phase 0 — Project Scope              COMPLETE
Phase 1 — Source/API Strategy        COMPLETE
Phase 2A — Count Audit               COMPLETE
Phase 2B — Raw Acquisition           COMPLETE
Phase 3 — Data Audit + Label Set     COMPLETE — labels LOCKED
Phase 4 — Modeling Dataset           COMPLETE
Phase 5 — Simple LSTM Baseline       COMPLETE
Phase 6 — Controlled Enhancements    NEXT
Phase 7 — DistilBERT Transfer Learn  PENDING
Phase 8 — Final Comparison           PENDING
Phase 9 — Documentation/Viva         PENDING
```

Phase 6 contains:

```text
E1 — BiLSTM
E2 — Stacked BiLSTM
E3 — GloVe
E4 — Regularization
E5 — Sequence Length
E6 — Class Weights
```

Phase 7 contains:

```text
E7 — DistilBERT
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

## Phase 3 findings carried forward

Audited 2026-08-18 over all 107,992 raw rows. Full report:
`reports/phase3_data_audit.md`. Decision: `reports/phase3_label_decision.md`.

**LABEL SET LOCKED — Option A: the 5 native CFPB `Product` values, used
verbatim.** No merging, renaming, or exclusion.

1. **The extract is clean.** 107,992 unique Complaint IDs, zero duplicates,
   zero missing or blank narratives, `Product` 100% populated, no unexpected
   product values, identical 16-column schema across all five files.
2. **Alternatives were rejected on evidence.** `Sub-product` is unusable
   (Debt collection's largest value is literally `I do not know` at 45.97%;
   Credit card and Student loan have only 2 sub-products each). `Issue` needs
   17 of its 48 values to cover 80% of rows with 16 values under 500 rows.
   `Sub-issue` is 100% missing for the whole Money transfer product.
3. **Class balance in this extract is an acquisition artifact.** 1.19x here
   against 8.3x in the real population. **E6 (class weights) should therefore
   be expected to produce a near-zero delta** — state that as a hypothesis
   before running it, or revisit the Phase 4 sampling strategy to preserve
   natural imbalance. This is a Phase 4 decision for the project owner.
4. **7,035 rows share exact narrative text** across 847 distinct texts
   (largest group 832 rows). These are distinct Complaint IDs with identical
   text, almost certainly complaint-mill boilerplate. Phase 4 must keep
   identical narratives on the same side of the train/test split or the
   evaluation is contaminated (section 27).
5. **7 duplicate narratives carry conflicting Product labels** — a small but
   real ceiling on achievable accuracy.
6. **29.07% of narratives exceed 256 words, 6.24% exceed 512.** These are
   whitespace word counts; the section 41 max-length fairness analysis must
   use real tokenizer output, which will truncate more.
7. **Products span different date ranges by design** (Debt collection 4
   months, Student loan 27), so a temporal split would not be comparable
   across classes. The stratified random split of section 26 stands.

---

# 83. Current Immediate Task

## Phase 6 — Controlled LSTM Enhancements

Phases 4 and 5 are complete. The modeling dataset `cfpb_phase4_v1` exists and
the E0 baseline is recorded in `reports/experiment_results.csv` at
**Macro-F1 0.8435 / Accuracy 84.45%**.

Phase 6 runs the recurrent enhancement ladder **on the local CPU**
(section 25A), in order:

```text
E1 — BiLSTM
E2 — Stacked BiLSTM
E3 — GloVe
E4 — Regularization
E5 — Sequence Length
E6 — Class Weights
```

Rules for every Phase 6 experiment:

1. change exactly one meaningful factor against its stated reference
2. keep every control in the section 43B table fixed
3. record the full section 43A structure — objective through KEEP/REJECT
4. append the actual result row to `reports/experiment_results.csv`
5. compute and report Δ Macro-F1
6. keep negative results (section 45)

E4 and E5 must remain **evidence-driven**: run them because the measured
behavior of an earlier experiment justifies them, not to add a row.

Do not begin Phase 7 (DistilBERT) until the best enhanced recurrent model has
been identified and justified.

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
