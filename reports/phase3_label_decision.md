# Phase 3 — Final Label-Set Decision

- Decided: 2026-08-18
- Evidence: [`reports/phase3_data_audit.md`](phase3_data_audit.md) — 107,992 raw CFPB complaints
- Gate: PROJECT_DIRECTIONS.md sections 11 and 23

---

## Decision

> **Final target label = 5 classes, using the native CFPB `Product` values directly (Option A).**

| # | Label | Rows | % of dataset |
|---|---|---:|---:|
| 1 | Debt collection | 24,007 | 22.23% |
| 2 | Checking or savings account | 21,547 | 19.95% |
| 3 | Money transfer, virtual currency, or money service | 21,437 | 19.85% |
| 4 | Credit card | 20,890 | 19.34% |
| 5 | Student loan | 20,111 | 18.62% |

No label is merged, renamed, split, or excluded. The CFPB `Product` string is
used verbatim as the target.

---

## Why Option A, from the data

### 1. The native labels are clean

Every integrity check the label set depends on passed on the real extract,
recomputed independently of the Phase 2B report:

- 107,992 rows, 107,992 unique Complaint IDs, **0 duplicates, 0 missing IDs**
- **0 missing and 0 blank narratives** — every row is usable for text classification
- `Product` is **100% populated**, with **no unexpected values** and none of the
  five expected products absent
- All five source files share an identical 16-column schema

There is no data-quality reason to derive a different taxonomy.

### 2. Every class has enough data

The smallest class, Student loan, has 20,111 rows. Against the project's
compute constraint (CPU-only, DistilBERT fine-tuned on a stratified ~8k–10k
subset), every class is comfortably large enough to train and evaluate on.
No class is anywhere near the very-sparse threshold.

### 3. The classes carry genuinely distinct vocabulary

The vocabulary fingerprint — words most over-represented in each product
relative to the rest of the corpus — separates cleanly along real product
lines rather than noise:

| Product | Distinctive vocabulary |
|---|---|
| Student loan | `mohela`, `pslf`, `forbearance`, `unsubsidized`, `navient`, `servicers`, `capitalization` |
| Money transfer | `zelle`, `paypal`, `moneygram`, `remitly`, `kraken`, `coinme`, `bitstamp`, `eth` |
| Debt collection | `validation`, `collection`, `alleged`, `mcm`, `tsi`, `sequium`, `original` |
| Checking or savings | `overdraft`, `atm`, `chexsystems`, `deposits`, `hsa` |
| Credit card | `mastercard`, `barclay`, `comenity`, `sapphire`, `merchant`, `interest` |

These are domain terms, servicer names, and product mechanics — exactly the
signal a text classifier should be able to learn. This is a bag-of-words proxy,
not proof a model will succeed, but it is strong evidence the classes are
separable from narrative text alone.

### 4. The alternatives are worse, and the audit shows why

**`Sub-product` rejected.** Internally inconsistent as a target:

- Debt collection's most common Sub-product is literally **`I do not know` at
  45.97%** — a non-category that would become the largest single class
- Credit card has only **2** Sub-products; Student loan only **2**
- Meanwhile Debt collection has 11 and Money transfer has 7

The granularity is wildly uneven across products, and the largest bucket is an
absence of information.

**`Issue` rejected.** 48 distinct values, and **17 are needed just to cover 80%**
of rows while **16 have under 500 rows**. That is a long-tailed 48-class problem
on a CPU-only budget — it would dominate the timeline and make macro-F1 hostage
to tiny classes, without serving the stated business problem (routing a
complaint to the right *product team*).

**`Sub-issue` rejected outright.** 134 distinct values, **19.96% missing overall**,
and decisively: **Sub-issue is 100% missing for the entire Money transfer
product**. A target undefined for a fifth of the dataset is not a viable target.

### 5. It matches the business problem

PROJECT_DIRECTIONS section 4 frames the task as routing a complaint to the
correct product team. `Product` *is* that routing key. Issue and Sub-issue
describe what went wrong, not who should handle it.

---

## Known limitations, recorded honestly

These do not change the decision, but each is a real property of this dataset
and must be carried into later phases rather than discovered during modeling.

### A. The class balance is an artifact of acquisition, not of reality

This extract is nearly balanced — **1.19x** between largest and smallest class.
That is a direct consequence of the Phase 2B rule, which targeted ~20,000 rows
per product. The true CFPB population is **8.3x** imbalanced (Debt collection
220,651 vs Student loan 26,534, per the Phase 2A audit).

**This has a direct consequence for experiment E3 (class weights).** The
project hypothesis is that class weighting improves minority-class performance.
On a 1.19x-balanced dataset, class weights should be expected to produce a
**near-zero delta** — there is barely any imbalance for them to correct.

That is a legitimate and explainable result, not a failure, and
PROJECT_DIRECTIONS section 45 explicitly permits negative findings. But it
should be stated as a hypothesis *before* E3 runs, not rationalised afterwards.
Two honest options exist, and this is a decision for the project owner at
Phase 4, not for the audit:

1. **Keep the balanced dataset** and report E3's null result, explaining that
   the acquisition strategy removed the imbalance the technique addresses.
2. **Construct the modeling dataset to preserve the population's natural
   imbalance** (down-sampling the larger classes proportionally rather than to
   a flat target), making E3 a meaningful test — at the cost of a smaller
   effective dataset.

### B. Duplicate narrative text is a leakage risk

**7,035 rows share exact narrative text with at least one other row**, across
847 distinct texts, with the largest single group at **832 rows** — almost
certainly complaint-mill boilerplate.

These are *not* duplicate Complaint IDs (there are none); they are distinct
complaints with identical text. If they are split randomly across train and
test, the model will be evaluated on text it memorised. Phase 4 must group
identical narratives onto the same side of the split
(PROJECT_DIRECTIONS section 27). Nothing was removed at the raw stage.

### C. A small cross-product ambiguity exists

Six credit-reporting-flavoured Issues appear under three different products
simultaneously (~4,505 rows total), for example `Incorrect information on your
report` under Checking or savings, Credit card, and Student loan. Narratives in
this group may read as credit-reporting complaints regardless of their product
label. Expect these in the confusion matrix and examine them during Phase 8
error analysis.

Separately, **7 duplicate narrative texts appear under more than one Product** —
identical text with conflicting labels. Tiny in volume, but a genuine label
ceiling: no model can classify both copies correctly.

### D. Narrative length will need a truncation decision

Median narrative is 168 words, but **29.07% exceed 256 words** and **6.24%
exceed 512**. These are whitespace word counts; DistilBERT's sub-word tokenizer
produces more tokens than words, so real truncation at a 256-*token* limit will
be higher. The `max_length` fairness analysis required by section 41 must use
actual tokenizer output, not these figures.

### E. Products cover different date ranges

By design: Debt collection spans 4 months (2026-03 → 2026-06) while Student
loan spans 27 (2024-04 → 2026-06). A temporal split would therefore not be
comparable across classes. The stratified random split specified in section 26
remains the right primary choice.

---

## Status

```
Phase 3 — Raw data audit          COMPLETE
Phase 3 — Label set               LOCKED: 5 native CFPB Product labels
Phase 4 — Dataset construction    NEXT
```

Per PROJECT_DIRECTIONS section 23 this decision is a gate: no model training
begins before it is documented. It now is.
