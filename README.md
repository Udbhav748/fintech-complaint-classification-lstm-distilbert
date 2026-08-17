# FinTech Complaint Classification: A Controlled LSTM-to-DistilBERT Enhancement Study

Classify real FinTech customer complaints into product categories, and measure
how much controlled recurrent-model enhancements improve that classification
compared with pretrained Transformer transfer learning.

The project question:

> How much can we improve FinTech complaint classification through controlled
> recurrent-model enhancements, and how does the resulting recurrent model
> compare with pretrained Transformer transfer learning?

The method is baseline first, then one controlled change at a time, recording
the macro-F1 delta and the reason for it at every step.

## Business problem

A FinTech company receives complaints as free text through several channels.
Each complaint has to reach the right product team. Manual routing does not
scale, so the routing decision is modelled as text classification:

```
Customer complaint text -> NLP classifier -> Complaint/product category
```

## Data source

The **CFPB Consumer Complaint Database**, a public database of real consumer
financial complaints, accessed through its official API.

- Source population: **2023-08-17 to 2026-08-17** (three years)
- Only complaints carrying a consumer narrative are in scope, since the task is
  text classification
- Provisional acquisition scope: five CFPB `Product` categories

The CFPB database is a project-selected dataset chosen because it provides real
FinTech complaint narratives. It is not mandated by the course material, and the
documentation does not claim otherwise.

## Project status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project definition and scope | Locked |
| 1 | Source / API strategy | Locked |
| 2A | CFPB count audit | **Complete** |
| 2B | Raw narrative acquisition | **Complete** |
| 3 | Raw data quality audit + final label set | Pending |
| 4 | Modelling dataset construction | Pending |
| 5 | LSTM baseline (E0) | Pending |
| 6 | Controlled LSTM enhancements (E1-E3) | Pending |
| 7 | DistilBERT transfer learning | Pending |
| 8 | Final comparison + error analysis | Pending |
| 9 | Documentation + demo | Pending |

## Phase 2A result

A single `/trends` request returns the full product-by-month breakdown for all
five products across the three-year window.

| Product | Narrative complaints | Median / month | Peak month |
|---|---:|---:|---:|
| Debt collection | 220,651 | 6,365 | 10,611 |
| Checking or savings account | 109,780 | 2,696 | 13,696 |
| Credit card | 108,204 | 2,979 | 4,065 |
| Money transfer, virtual currency, or money service | 86,988 | 944 | 41,618 |
| Student loan | 26,534 | 728 | 1,828 |

**Total: 552,157 narrative-bearing complaints**, with no product falling below
the very-sparse threshold. Three findings carried into Phase 2B:

1. **2026-07 is incomplete.** Volume collapses across every product because the
   CFPB publishes a narrative only after the company responds. The effective end
   of the source population is earlier than the nominal 2026-08-17.
2. **Money transfer 2025-01 is a 44x spike** (41,618 against a median of 944).
   One month would otherwise dominate that class and distort macro-F1.
3. **`has_narrative=yes` does not work.** The published API spec is wrong; the
   live endpoint returns HTTP 424 and the working value is `true`.

Full detail: [`reports/phase2a_count_audit.md`](reports/phase2a_count_audit.md).

## Running the count audit

```bash
pip install -r requirements.txt
python -m src.data_acquisition.cfpb_counts --config configs/data.yaml
```

Outputs land in `reports/`:

- `phase2a_product_month_counts.csv` - one row per product-month
- `phase2a_count_audit.md` - audit report with review flags
- `phase2a_trends_raw.json` - unmodified API response

## Phase 2B plan

The extract targets a working dataset of roughly 80,000-120,000 rows rather than
the full 552,157-row population. The selection rule is deterministic: for each
product, take the most recent complete month and walk backwards, adding whole
calendar months until that product's Phase 2A count reaches 20,000 rows.

| Product | Windows | Months | Expected rows |
|---|---:|---|---:|
| Debt collection | 4 | 2026-03 → 2026-06 | 24,007 |
| Checking or savings account | 7 | 2025-12 → 2026-06 | 21,547 |
| Money transfer, virtual currency, or money service | 13 | 2025-06 → 2026-06 | 21,437 |
| Credit card | 7 | 2025-12 → 2026-06 | 20,890 |
| Student loan | 27 | 2024-04 → 2026-06 | 20,111 |
| **Total** | **58** | | **107,992** |

Windows are monthly and non-overlapping. Acquisition ends at **2026-06**, not
the nominal 2026-08-17, because Phase 2A showed 2026-07 is still filling in.

The recency rule also happens to exclude both volume spikes Phase 2A flagged
(Money transfer 2025-01 at 44x its median, Checking or savings 2025-01 at 5.1x).
That is a consequence of selecting recent windows, not a removal step.

```bash
python -m src.data_acquisition.cfpb_downloader --dry-run   # show the plan
python -m src.data_acquisition.cfpb_downloader             # acquire
python -m src.data_acquisition.cfpb_downloader --verify    # validate files on disk
```

`--verify` exists because the extract may be obtained outside this script. It
runs the same integrity checks over whatever is already under
`data/raw/cfpb/source/` and compares coverage per product-month against the
Phase 2A counts, so a manually exported extract is validated exactly as
strictly as a scripted one. Coverage is checked against the data itself rather
than against request boundaries, so the file layout does not have to match the
monthly windows.

Each product-month is written as an immutable CSV under
`data/raw/cfpb/source/`, with an audit trail in
`data/raw/cfpb/manifests/acquisition_manifest.json` recording the query, the
expected and retrieved row counts, checksums, and any failed windows.

**Result:** 107,992 rows acquired. The CFPB edge began returning HTTP 403 to
this client across the whole domain before the scripted per-month acquisition
could run, so the same five product windows were retrieved manually via the
official search CSV export and validated with `--verify` (below) instead. All
58 planned product-months matched their Phase 2A expected count exactly - zero
delta everywhere, no duplicate Complaint IDs, no missing narratives. Full
detail: [`reports/phase2b_acquisition_report.md`](reports/phase2b_acquisition_report.md).

## Project structure

```
configs/          Experiment and data configuration
data/raw/         Official CFPB source data (not committed)
data/interim/     Audited and transformed data
data/processed/   Final train/validation/test sets
src/              Reusable implementation code
notebooks/        Exploration and result presentation
reports/          Audit reports, experiment registry, figures
tests/            Tests
```

`PROJECT_DIRECTIONS.md` is the single source of truth for scope, methodology,
and coding standards. Read it before changing anything.

## Compute constraint

Development is **CPU-only** (no CUDA device available), which is a real
constraint on the experiment design rather than a methodological choice:

- LSTM experiments use the larger modelling dataset
- DistilBERT fine-tuning uses a stratified ~8,000-10,000 sample subset
- DistilBERT runs 1-2 epochs initially, with early stopping

This is documented honestly in the final report. Results are never extrapolated
to a configuration that was not actually run.

## Evaluation

Primary metric is **macro-F1**, because the class distribution is imbalanced and
accuracy would hide minority-class failure. Accuracy, per-class precision and
recall, confusion matrices, training time and parameter counts are recorded
alongside it.

The headline comparison is Simple LSTM vs Best Enhanced LSTM vs Fine-tuned
DistilBERT, evaluated on the same split with the same protocol.
