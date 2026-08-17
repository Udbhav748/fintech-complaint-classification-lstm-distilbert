# Phase 2A - CFPB Count Audit

- Retrieved: 2026-08-18T00:54:53+05:30
- Source population: 2023-08-17 to 2026-08-17
- Narrative filter: has_narrative=true
- Products audited: 5
- Product-months observed: 180

## Per-product totals (narrative-bearing complaints)

| product                                            |   total_complaints |   months_observed |   min_month_count |   median_month_count |   max_month_count | is_very_sparse_overall   |
|:---------------------------------------------------|-------------------:|------------------:|------------------:|---------------------:|------------------:|:-------------------------|
| Debt collection                                    |             220651 |                36 |               798 |               6365   |             10611 | False                    |
| Checking or savings account                        |             109780 |                36 |               705 |               2695.5 |             13696 | False                    |
| Credit card                                        |             108204 |                36 |               624 |               2978.5 |              4065 | False                    |
| Money transfer, virtual currency, or money service |              86988 |                36 |               116 |                944   |             41618 | False                    |
| Student loan                                       |              26534 |                36 |                17 |                728   |              1828 | False                    |

## Derived acquisition windows

Window sizes are computed from each product's peak observed month against the oversized threshold of 80,000 complaints per request, rather than from an assumed monthly average.

| product                                            |   peak_month_count |   median_month_count | recommended_window   |   months_per_request |   estimated_requests |
|:---------------------------------------------------|-------------------:|---------------------:|:---------------------|---------------------:|---------------------:|
| Debt collection                                    |              10611 |               6365   | 7-month              |                    7 |                    6 |
| Checking or savings account                        |              13696 |               2695.5 | 5-month              |                    5 |                    8 |
| Credit card                                        |               4065 |               2978.5 | 19-month             |                   19 |                    2 |
| Money transfer, virtual currency, or money service |              41618 |                944   | 1-month              |                    1 |                   36 |
| Student loan                                       |               1828 |                728   | 36-month             |                   36 |                    1 |

## Review flags

- Sparse product-months (< 500): 7
- Oversized product-months (> 80,000): 0
- Very sparse products (< 5,000 total): 0

Flags are review markers. No product or month is removed on the basis of a flag; the label set is decided in Phase 3 after the raw data audit.

## Incomplete trailing months

Volume collapses across every product in the following month(s), which indicates the CFPB publication lag rather than a genuine drop in complaints. A narrative is only published after the company responds, so the most recent months are still filling in.

- 2026-07

Recommendation: treat the source population as ending before 2026-07 for Phase 2B acquisition, and record the effective end date in the acquisition manifest.

## Volume anomalies

Product-months exceeding five times that product's median volume. A single dominant month can skew the class distribution and therefore macro-F1, so these require a sampling decision in Phase 4.

| product                                            | month   |   complaint_count |   product_median |   times_median |
|:---------------------------------------------------|:--------|------------------:|-----------------:|---------------:|
| Money transfer, virtual currency, or money service | 2025-01 |             41618 |            944   |           44.1 |
| Checking or savings account                        | 2025-01 |             13696 |           2695.5 |            5.1 |

### Sparse product-months

| product                                            | month   |   complaint_count |
|:---------------------------------------------------|:--------|------------------:|
| Money transfer, virtual currency, or money service | 2023-08 |               323 |
| Money transfer, virtual currency, or money service | 2026-07 |               116 |
| Student loan                                       | 2023-08 |               244 |
| Student loan                                       | 2024-11 |               481 |
| Student loan                                       | 2026-02 |               389 |
| Student loan                                       | 2026-06 |               355 |
| Student loan                                       | 2026-07 |                17 |
