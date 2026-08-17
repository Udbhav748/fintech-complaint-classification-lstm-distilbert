# Raw CFPB Source Data

This directory holds the immutable raw extract produced by **Phase 2B — Raw
Narrative Acquisition**. Nothing here has been cleaned, deduplicated, merged,
balanced, or tokenized.

## Dataset source

**CFPB Consumer Complaint Database**
<https://www.consumerfinance.gov/data-research/consumer-complaints/>

> The CFPB Consumer Complaint Database is a project-selected real-world dataset
> aligned with the FinTech complaint-classification use case from the FWC
> material.

The FWC material establishes the complaint-classification business case. It does
**not** mandate the CFPB dataset. Choosing CFPB is a project decision, made
because it provides real financial complaint narratives suitable for supervised
text classification.

## Retrieval method

Date-windowed CSV requests against the public search API:

```
https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/
    ?product=<product>
    &has_narrative=true
    &date_received_min=<window start>
    &date_received_max=<window end>
    &format=csv
    &no_aggs=true
    &size=60000
```

One request per product-month, one immutable CSV per product-month. Windows are
calendar months and do not overlap.

Two behaviours were established against the live API during Phase 2A and are
relied on here:

- `has_narrative=true` works; the `yes`/`no` values in the published swagger
  spec return HTTP 424.
- A custom `User-Agent` header triggers the CDN WAF (HTTP 403), so the default
  client User-Agent is left in place.

The CFPB edge (Akamai) also rate-limits bursts and answers HTTP 403 for a period
afterwards, so requests are spaced and retried with backoff.

## Source time range

| | |
|---|---|
| Nominal source population | 2023-08-17 → 2026-08-17 |
| Effective acquisition end | 2026-06-30 |

Phase 2A found that complaint volume collapses across every product in
**2026-07**. The CFPB publishes a narrative only after the company responds, so
recent months are still filling in. That month is therefore excluded, and the
effective end date is recorded in the acquisition manifest.

## Selected product categories

These are **provisional acquisition categories, not the final ML label set**.
The label set is locked in Phase 3, after the raw data quality audit.

1. `Debt collection`
2. `Credit card`
3. `Checking or savings account`
4. `Money transfer, virtual currency, or money service`
5. `Student loan`

Credit reporting products are excluded from the initial scope because their
volume is orders of magnitude larger than the other candidates, and macro-F1 is
the project's primary metric — a dominant class would weaken the interpretation
of minority-class performance. This is an experimental-scope decision, not a
claim that credit reporting is outside FinTech.

## Narrative-only filter

Only complaints carrying a consumer narrative are usable for text
classification, so `has_narrative=true` is applied at the source. The full
source population under this filter is **552,157** complaints across the five
products.

## Which windows were acquired, and why

The goal was a working extract of roughly 80,000–120,000 rows, not the full
552,157-row population.

The selection rule is deterministic: for each product, take the most recent
complete month and walk backwards, adding whole months until that product's
cumulative Phase 2A count reaches **20,000** rows. The month that crosses the
target is included.

| Product | Windows | Months | Expected rows |
|---|---:|---|---:|
| Debt collection | 4 | 2026-03 → 2026-06 | 24,007 |
| Checking or savings account | 7 | 2025-12 → 2026-06 | 21,547 |
| Money transfer, virtual currency, or money service | 13 | 2025-06 → 2026-06 | 21,437 |
| Credit card | 7 | 2025-12 → 2026-06 | 20,890 |
| Student loan | 27 | 2024-04 → 2026-06 | 20,111 |
| **Total** | **58** | | **107,992** |

Expected row counts come from the Phase 2A trends counts. The manifest records
the count actually retrieved for every window alongside its expected count, so
coverage is verifiable rather than assumed.

## File structure

```
data/raw/cfpb/
├── source/
│   └── <product_slug>/
│       └── <product_slug>_<YYYY-MM>.csv
├── manifests/
│   └── acquisition_manifest.json
└── README.md
```

Raw CSVs are **not committed to git** (see `.gitignore`). They are reproducible
from the manifest and the acquisition script:

```
python -m src.data_acquisition.cfpb_downloader --dry-run   # show the plan
python -m src.data_acquisition.cfpb_downloader             # acquire
```

## Preserved fields

All columns returned by the API are preserved. At minimum the extract retains
Complaint ID, Date received, Product, Sub-product, Issue, Sub-issue, Consumer
complaint narrative, Company, State, ZIP code, Submitted via, Company response
to consumer, Timely response, Company public response, and Date sent to company.

## Known limitations

- **Publication lag.** A narrative appears only after the company responds, so
  the most recent months under-report. 2026-07 is excluded for this reason;
  2026-06 may still be slightly incomplete.
- **Recency bias.** The extract is a recent slice, not a random sample of the
  three-year population. It is deliberately weighted toward current complaint
  language.
- **Unequal time spans.** Because the row target is per product, the products
  cover different date ranges (4 months for Debt collection, 27 for Student
  loan). Any temporal analysis must account for this.
- **Volume spikes excluded by recency.** Phase 2A flagged two anomalous months —
  Money transfer 2025-01 (41,618, 44x its median) and Checking or savings
  2025-01 (13,696, 5.1x). Neither falls inside the acquired windows, so neither
  distorts this extract. This is a consequence of the recency rule, not a
  removal step.
- **Self-selected complainants.** CFPB complaints are submitted voluntarily and
  are not a representative sample of all consumer financial problems.
- **Duplicates are not removed.** Duplicate Complaint IDs are counted and
  reported, never dropped, at the raw stage.
