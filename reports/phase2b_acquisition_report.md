# Phase 2B - Raw Extract Verification

- Checked: 2026-08-18T01:54:49+05:30
- Files inspected: 5

## Integrity checks

- Total rows: 107,992
- Date received range: 2024-04-01 to 2026-06-30
- Missing narratives: 0
- Duplicate Complaint IDs: 0
- Unique Complaint IDs: 107,992

### Rows per Product

| product                                            |   rows |
|:---------------------------------------------------|-------:|
| Debt collection                                    |  24007 |
| Checking or savings account                        |  21547 |
| Money transfer, virtual currency, or money service |  21437 |
| Credit card                                        |  20890 |
| Student loan                                       |  20111 |

## Files

| file                                               |   rows |   size_bytes | sha256              |
|:---------------------------------------------------|-------:|-------------:|:--------------------|
| checking_or_savings_account_2025-12_to_2026-06.csv |  21547 |     36433130 | 0e96fa84631c944e... |
| credit_card_2025-12_to_2026-06.csv                 |  20890 |     38172585 | 77c549cd728e4176... |
| debt_collection_2026-03_to_2026-06.csv             |  24007 |     32557206 | 7a09eafac047ccc5... |
| money_transfer_2025-06_to_2026-06.csv              |  21437 |     32445934 | d0c606bc788d6563... |
| student_loan_2024-04_to_2026-06.csv                |  20111 |     31950976 | 594cab36296bdaca... |

## Coverage against the Phase 2A counts

- Product-months planned: 58
- Product-months missing: 0
- Product-months outside the plan: 0

| product                                            | month   |   expected_rows |   retrieved_rows |   row_delta | note   |
|:---------------------------------------------------|:--------|----------------:|-----------------:|------------:|:-------|
| Checking or savings account                        | 2025-12 |            3215 |             3215 |           0 |        |
| Checking or savings account                        | 2026-01 |            3792 |             3792 |           0 |        |
| Checking or savings account                        | 2026-02 |            1978 |             1978 |           0 |        |
| Checking or savings account                        | 2026-03 |            3448 |             3448 |           0 |        |
| Checking or savings account                        | 2026-04 |            3774 |             3774 |           0 |        |
| Checking or savings account                        | 2026-05 |            3004 |             3004 |           0 |        |
| Checking or savings account                        | 2026-06 |            2336 |             2336 |           0 |        |
| Credit card                                        | 2025-12 |            3183 |             3183 |           0 |        |
| Credit card                                        | 2026-01 |            3788 |             3788 |           0 |        |
| Credit card                                        | 2026-02 |            1983 |             1983 |           0 |        |
| Credit card                                        | 2026-03 |            3357 |             3357 |           0 |        |
| Credit card                                        | 2026-04 |            3599 |             3599 |           0 |        |
| Credit card                                        | 2026-05 |            2958 |             2958 |           0 |        |
| Credit card                                        | 2026-06 |            2022 |             2022 |           0 |        |
| Debt collection                                    | 2026-03 |            8249 |             8249 |           0 |        |
| Debt collection                                    | 2026-04 |            6638 |             6638 |           0 |        |
| Debt collection                                    | 2026-05 |            5520 |             5520 |           0 |        |
| Debt collection                                    | 2026-06 |            3600 |             3600 |           0 |        |
| Money transfer, virtual currency, or money service | 2025-06 |            1670 |             1670 |           0 |        |
| Money transfer, virtual currency, or money service | 2025-07 |            1932 |             1932 |           0 |        |
| Money transfer, virtual currency, or money service | 2025-08 |            1920 |             1920 |           0 |        |
| Money transfer, virtual currency, or money service | 2025-09 |            1634 |             1634 |           0 |        |
| Money transfer, virtual currency, or money service | 2025-10 |            2634 |             2634 |           0 |        |
| Money transfer, virtual currency, or money service | 2025-11 |            1673 |             1673 |           0 |        |
| Money transfer, virtual currency, or money service | 2025-12 |            1497 |             1497 |           0 |        |
| Money transfer, virtual currency, or money service | 2026-01 |            1767 |             1767 |           0 |        |
| Money transfer, virtual currency, or money service | 2026-02 |             981 |              981 |           0 |        |
| Money transfer, virtual currency, or money service | 2026-03 |            1662 |             1662 |           0 |        |
| Money transfer, virtual currency, or money service | 2026-04 |            1842 |             1842 |           0 |        |
| Money transfer, virtual currency, or money service | 2026-05 |            1397 |             1397 |           0 |        |
| Money transfer, virtual currency, or money service | 2026-06 |             828 |              828 |           0 |        |
| Student loan                                       | 2024-04 |             632 |              632 |           0 |        |
| Student loan                                       | 2024-05 |             731 |              731 |           0 |        |
| Student loan                                       | 2024-06 |             584 |              584 |           0 |        |
| Student loan                                       | 2024-07 |             540 |              540 |           0 |        |
| Student loan                                       | 2024-08 |             520 |              520 |           0 |        |
| Student loan                                       | 2024-09 |             758 |              758 |           0 |        |
| Student loan                                       | 2024-10 |             726 |              726 |           0 |        |
| Student loan                                       | 2024-11 |             481 |              481 |           0 |        |
| Student loan                                       | 2024-12 |             746 |              746 |           0 |        |
| Student loan                                       | 2025-01 |            1209 |             1209 |           0 |        |
| Student loan                                       | 2025-02 |             992 |              992 |           0 |        |
| Student loan                                       | 2025-03 |            1828 |             1828 |           0 |        |
| Student loan                                       | 2025-04 |            1148 |             1148 |           0 |        |
| Student loan                                       | 2025-05 |            1054 |             1054 |           0 |        |
| Student loan                                       | 2025-06 |             765 |              765 |           0 |        |
| Student loan                                       | 2025-07 |             848 |              848 |           0 |        |
| Student loan                                       | 2025-08 |             675 |              675 |           0 |        |
| Student loan                                       | 2025-09 |             658 |              658 |           0 |        |
| Student loan                                       | 2025-10 |             556 |              556 |           0 |        |
| Student loan                                       | 2025-11 |             587 |              587 |           0 |        |
| Student loan                                       | 2025-12 |             554 |              554 |           0 |        |
| Student loan                                       | 2026-01 |             764 |              764 |           0 |        |
| Student loan                                       | 2026-02 |             389 |              389 |           0 |        |
| Student loan                                       | 2026-03 |             703 |              703 |           0 |        |
| Student loan                                       | 2026-04 |             764 |              764 |           0 |        |
| Student loan                                       | 2026-05 |             544 |              544 |           0 |        |
| Student loan                                       | 2026-06 |             355 |              355 |           0 |        |
