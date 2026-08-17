# Phase 3 - Raw Data Quality Audit

- Generated: 2026-08-18T02:08:34+05:30
- Source: `data/raw/cfpb/source`
- Rows audited: 107,992

This report measures the raw extract exactly as acquired. No cleaning, deduplication, label merging, balancing, or tokenization was performed, and no file under `data/raw/` was modified.

---

## 1. File and schema inventory

| file                                               |   rows |   columns | missing_expected_columns   |   size_mb |
|:---------------------------------------------------|-------:|----------:|:---------------------------|----------:|
| checking_or_savings_account_2025-12_to_2026-06.csv |  21547 |        16 | none                       |      34.7 |
| credit_card_2025-12_to_2026-06.csv                 |  20890 |        16 | none                       |      36.4 |
| debt_collection_2026-03_to_2026-06.csv             |  24007 |        16 | none                       |      31   |
| money_transfer_2025-06_to_2026-06.csv              |  21437 |        16 | none                       |      30.9 |
| student_loan_2024-04_to_2026-06.csv                |  20111 |        16 | none                       |      30.5 |

All files share an identical schema, so the combined audit view needs no column reconciliation.

---

## 2. Combined dataset inventory

Independently recomputed here rather than carried over from the Phase 2B report.

- Total rows: 107,992
- Unique Complaint IDs: 107,992
- Duplicate Complaint IDs: 0
- Missing Complaint IDs: 0
- Date received range: 2024-04-01 to 2026-06-30
- Unparseable dates: 0
- Missing narratives: 0
- Blank/whitespace-only narratives: 0
- Source files: 5
- Unexpected Product values: none
- Expected Products absent: none

### Missing values by field

| field                        |   missing |   missing_pct |
|:-----------------------------|----------:|--------------:|
| Complaint ID                 |         0 |          0    |
| Date received                |         0 |          0    |
| Product                      |         0 |          0    |
| Sub-product                  |         0 |          0    |
| Issue                        |         0 |          0    |
| Sub-issue                    |     21551 |         19.96 |
| Consumer complaint narrative |         0 |          0    |
| Company                      |         0 |          0    |
| State                        |      1318 |          1.22 |
| ZIP code                     |       263 |          0.24 |
| Submitted via                |         0 |          0    |
| Company response to consumer |         0 |          0    |
| Timely response?             |         0 |          0    |
| Company public response      |     74249 |         68.75 |
| Date sent to company         |         0 |          0    |

---

## 3. Product label audit

| product                                            |   count |   pct_of_dataset |   months_observed |   min_month |   median_month |   max_month | first_month   | last_month   |
|:---------------------------------------------------|--------:|-----------------:|------------------:|------------:|---------------:|------------:|:--------------|:-------------|
| Debt collection                                    |   24007 |            22.23 |                 4 |        3600 |           6079 |        8249 | 2026-03       | 2026-06      |
| Checking or savings account                        |   21547 |            19.95 |                 7 |        1978 |           3215 |        3792 | 2025-12       | 2026-06      |
| Money transfer, virtual currency, or money service |   21437 |            19.85 |                13 |         828 |           1670 |        2634 | 2025-06       | 2026-06      |
| Credit card                                        |   20890 |            19.34 |                 7 |        1983 |           3183 |        3788 | 2025-12       | 2026-06      |
| Student loan                                       |   20111 |            18.62 |                27 |         355 |            703 |        1828 | 2024-04       | 2026-06      |

- Candidate classes: 5
- Largest: Debt collection (24,007)
- Smallest: Student loan (20,111)
- Imbalance ratio: **1.19x**

The near-even distribution is a direct consequence of the Phase 2B acquisition rule, which targeted roughly 20,000 rows per product. It is a property of this extract, not of the CFPB population, and the Phase 2A audit records the true population imbalance (Debt collection 220,651 against Student loan 26,534, an 8.3x ratio).

---

## 4. Sub-product audit

### Structure within each Product

| product                                            |   distinct_sub_product | top_value                                  |   top_value_pct |   top3_pct |   categories_under_1pct |   missing_pct |
|:---------------------------------------------------|-----------------------:|:-------------------------------------------|----------------:|-----------:|------------------------:|--------------:|
| Checking or savings account                        |                      4 | Checking account                           |           83.07 |      98.84 |                       0 |             0 |
| Credit card                                        |                      2 | General-purpose credit card or charge card |           86.55 |     100    |                       0 |             0 |
| Debt collection                                    |                     11 | I do not know                              |           45.97 |      76.56 |                       3 |             0 |
| Money transfer, virtual currency, or money service |                      7 | Mobile or digital wallet                   |           47.06 |      88.1  |                       1 |             0 |
| Student loan                                       |                      2 | Federal student loan servicing             |           77.33 |     100    |                       0 |             0 |

### Top Sub-products per Product

| product                                            | Sub-product                                      |   count |   pct_within_product |
|:---------------------------------------------------|:-------------------------------------------------|--------:|---------------------:|
| Checking or savings account                        | Checking account                                 |   17900 |                83.07 |
| Checking or savings account                        | Other banking product or service                 |    1928 |                 8.95 |
| Checking or savings account                        | Savings account                                  |    1468 |                 6.81 |
| Checking or savings account                        | CD (Certificate of Deposit)                      |     251 |                 1.16 |
| Credit card                                        | General-purpose credit card or charge card       |   18080 |                86.55 |
| Credit card                                        | Store credit card                                |    2810 |                13.45 |
| Debt collection                                    | I do not know                                    |   11036 |                45.97 |
| Debt collection                                    | Credit card debt                                 |    4046 |                16.85 |
| Debt collection                                    | Other debt                                       |    3297 |                13.73 |
| Debt collection                                    | Rental debt                                      |    1745 |                 7.27 |
| Debt collection                                    | Telecommunications debt                          |    1607 |                 6.69 |
| Debt collection                                    | Medical debt                                     |     920 |                 3.83 |
| Debt collection                                    | Auto debt                                        |     707 |                 2.94 |
| Debt collection                                    | Payday loan debt                                 |     386 |                 1.61 |
| Debt collection                                    | Private student loan debt                        |     103 |                 0.43 |
| Debt collection                                    | Mortgage debt                                    |      82 |                 0.34 |
| Debt collection                                    | Federal student loan debt                        |      78 |                 0.32 |
| Money transfer, virtual currency, or money service | Mobile or digital wallet                         |   10088 |                47.06 |
| Money transfer, virtual currency, or money service | Domestic (US) money transfer                     |    5702 |                26.6  |
| Money transfer, virtual currency, or money service | Virtual currency                                 |    3097 |                14.45 |
| Money transfer, virtual currency, or money service | International money transfer                     |    1483 |                 6.92 |
| Money transfer, virtual currency, or money service | Money order, traveler's check or cashier's check |     545 |                 2.54 |
| Money transfer, virtual currency, or money service | Check cashing service                            |     327 |                 1.53 |
| Money transfer, virtual currency, or money service | Foreign currency exchange                        |     195 |                 0.91 |
| Student loan                                       | Federal student loan servicing                   |   15551 |                77.33 |
| Student loan                                       | Private student loan                             |    4560 |                22.67 |

---

## 5. Issue audit

- Distinct Issue values: 48
- Missing Issue: 0 (0.0%)
- Top 10 Issues cover 64.98% of rows
- Issues needed to cover 80% of rows: 17
- Issues with under 500 rows: 16

### Structure within each Product

| product                                            |   distinct_issue | top_value                                       |   top_value_pct |   top3_pct |   categories_under_1pct |   missing_pct |
|:---------------------------------------------------|-----------------:|:------------------------------------------------|----------------:|-----------:|------------------------:|--------------:|
| Checking or savings account                        |               11 | Managing an account                             |           56.91 |      85.93 |                       6 |             0 |
| Credit card                                        |               15 | Problem with a purchase shown on your statement |           34.64 |      59.01 |                       4 |             0 |
| Debt collection                                    |                7 | Attempts to collect debt not owed               |           52.61 |      83.13 |                       0 |             0 |
| Money transfer, virtual currency, or money service |               16 | Fraud or scam                                   |           28.46 |      65.45 |                       6 |             0 |
| Student loan                                       |               11 | Dealing with your lender or servicer            |           62.09 |      89.26 |                       5 |             0 |

### Top Issues per Product

| product                                            | Issue                                                           |   count |   pct_within_product |
|:---------------------------------------------------|:----------------------------------------------------------------|--------:|---------------------:|
| Checking or savings account                        | Managing an account                                             |   12263 |                56.91 |
| Checking or savings account                        | Problem with a lender or other company charging your account    |    3342 |                15.51 |
| Checking or savings account                        | Closing an account                                              |    2911 |                13.51 |
| Checking or savings account                        | Opening an account                                              |    1530 |                 7.1  |
| Checking or savings account                        | Problem caused by your funds being low                          |    1362 |                 6.32 |
| Checking or savings account                        | Problem with a company's investigation into an existing problem |      47 |                 0.22 |
| Checking or savings account                        | Incorrect information on your report                            |      47 |                 0.22 |
| Checking or savings account                        | Problem with fraud alerts or security freezes                   |      20 |                 0.09 |
| Checking or savings account                        | Credit monitoring or identity theft protection services         |      11 |                 0.05 |
| Checking or savings account                        | Improper use of your report                                     |       9 |                 0.04 |
| Checking or savings account                        | Unable to get your credit report or credit score                |       5 |                 0.02 |
| Credit card                                        | Problem with a purchase shown on your statement                 |    7236 |                34.64 |
| Credit card                                        | Other features, terms, or problems                              |    2586 |                12.38 |
| Credit card                                        | Fees or interest                                                |    2506 |                12    |
| Credit card                                        | Getting a credit card                                           |    2064 |                 9.88 |
| Credit card                                        | Closing your account                                            |    1527 |                 7.31 |
| Credit card                                        | Problem when making payments                                    |    1104 |                 5.28 |
| Credit card                                        | Advertising and marketing, including promotional offers         |    1083 |                 5.18 |
| Credit card                                        | Trouble using your card                                         |     941 |                 4.5  |
| Credit card                                        | Incorrect information on your report                            |     874 |                 4.18 |
| Credit card                                        | Problem with a company's investigation into an existing problem |     510 |                 2.44 |
| Credit card                                        | Struggling to pay your bill                                     |     253 |                 1.21 |
| Credit card                                        | Improper use of your report                                     |     113 |                 0.54 |
| Credit card                                        | Credit monitoring or identity theft protection services         |      53 |                 0.25 |
| Credit card                                        | Problem with fraud alerts or security freezes                   |      33 |                 0.16 |
| Credit card                                        | Unable to get your credit report or credit score                |       7 |                 0.03 |
| Debt collection                                    | Attempts to collect debt not owed                               |   12631 |                52.61 |
| Debt collection                                    | Written notification about debt                                 |    4053 |                16.88 |
| Debt collection                                    | False statements or representation                              |    3273 |                13.63 |
| Debt collection                                    | Took or threatened to take negative or legal action             |    2189 |                 9.12 |
| Debt collection                                    | Communication tactics                                           |    1061 |                 4.42 |
| Debt collection                                    | Electronic communications                                       |     499 |                 2.08 |
| Debt collection                                    | Threatened to contact someone or share information improperly   |     301 |                 1.25 |
| Money transfer, virtual currency, or money service | Fraud or scam                                                   |    6100 |                28.46 |
| Money transfer, virtual currency, or money service | Other transaction problem                                       |    4614 |                21.52 |
| Money transfer, virtual currency, or money service | Unauthorized transactions or other transaction problem          |    3317 |                15.47 |
| Money transfer, virtual currency, or money service | Trouble accessing funds in your mobile or digital wallet        |    2237 |                10.44 |
| Money transfer, virtual currency, or money service | Money was not available when promised                           |    1583 |                 7.38 |
| Money transfer, virtual currency, or money service | Managing, opening, or closing your mobile wallet account        |    1074 |                 5.01 |
| Money transfer, virtual currency, or money service | Other service problem                                           |     565 |                 2.64 |
| Money transfer, virtual currency, or money service | Confusing or missing disclosures                                |     557 |                 2.6  |
| Money transfer, virtual currency, or money service | Problem with customer service                                   |     362 |                 1.69 |
| Money transfer, virtual currency, or money service | Unexpected or other fees                                        |     354 |                 1.65 |
| Money transfer, virtual currency, or money service | Wrong amount charged or received                                |     194 |                 0.9  |
| Money transfer, virtual currency, or money service | Confusing or misleading advertising or marketing                |     164 |                 0.77 |
| Money transfer, virtual currency, or money service | Lost or stolen money order                                      |     154 |                 0.72 |
| Money transfer, virtual currency, or money service | Problem adding money                                            |     109 |                 0.51 |
| Money transfer, virtual currency, or money service | Overdraft, savings, or rewards features                         |      39 |                 0.18 |
| Student loan                                       | Dealing with your lender or servicer                            |   12487 |                62.09 |
| Student loan                                       | Struggling to repay your loan                                   |    4126 |                20.52 |
| Student loan                                       | Incorrect information on your report                            |    1339 |                 6.66 |
| Student loan                                       | Improper use of your report                                     |     754 |                 3.75 |
| Student loan                                       | Getting a loan                                                  |     517 |                 2.57 |
| Student loan                                       | Problem with a company's investigation into an existing problem |     490 |                 2.44 |
| Student loan                                       | Credit monitoring or identity theft protection services         |     115 |                 0.57 |
| Student loan                                       | Issue where my lender is my school                              |     104 |                 0.52 |
| Student loan                                       | Issue with income share agreement                               |     101 |                 0.5  |
| Student loan                                       | Problem with fraud alerts or security freezes                   |      61 |                 0.3  |
| Student loan                                       | Unable to get your credit report or credit score                |      17 |                 0.08 |

### Issues appearing under more than one Product

| Issue                                                           | products                                               |   product_count |   rows |
|:----------------------------------------------------------------|:-------------------------------------------------------|----------------:|-------:|
| Incorrect information on your report                            | Checking or savings account, Credit card, Student loan |               3 |   2260 |
| Problem with a company's investigation into an existing problem | Checking or savings account, Credit card, Student loan |               3 |   1047 |
| Improper use of your report                                     | Checking or savings account, Credit card, Student loan |               3 |    876 |
| Credit monitoring or identity theft protection services         | Checking or savings account, Credit card, Student loan |               3 |    179 |
| Problem with fraud alerts or security freezes                   | Checking or savings account, Credit card, Student loan |               3 |    114 |
| Unable to get your credit report or credit score                | Checking or savings account, Credit card, Student loan |               3 |     29 |

---

## 6. Sub-issue audit

- Distinct Sub-issue values: 134
- Missing Sub-issue: 21,551 (19.96%)
- Sub-issues needed to cover 80% of rows: 34
- Sub-issues with under 100 rows: 50
- Sub-issues with under 500 rows: 90

### Structure within each Product

| product                                            |   distinct_sub_issue | top_value                                                                        |   top_value_pct |   top3_pct |   categories_under_1pct |   missing_pct |
|:---------------------------------------------------|---------------------:|:---------------------------------------------------------------------------------|----------------:|-----------:|------------------------:|--------------:|
| Checking or savings account                        |                   46 | Deposits and withdrawals                                                         |           19.96 |      46.18 |                      29 |          0.09 |
| Credit card                                        |                   55 | Credit card company isn't resolving a dispute about a purchase on your statement |           26.72 |      41.2  |                      34 |          0.16 |
| Debt collection                                    |                   29 | Debt is not yours                                                                |           30.64 |      59.21 |                      17 |          0    |
| Money transfer, virtual currency, or money service |                    1 | (missing)                                                                        |          100    |     100    |                       0 |        100    |
| Student loan                                       |                   52 | Trouble with how payments are being handled                                      |           20.96 |      52.19 |                      37 |          0.3  |

---

## 7. Narrative quality audit

- Rows with a narrative: 107,992
- Missing narrative: 0
- Blank/whitespace-only: 0

### Character length

|   min |   p25 |   median |   mean |    std |   p75 |   p90 |   p95 |   p99 |   max |
|------:|------:|---------:|-------:|-------:|------:|------:|------:|------:|------:|
|    32 |   529 |      988 | 1307.1 | 1284.1 |  1689 |  2588 |  3385 |  5929 | 31995 |

### Word length (whitespace split; exploratory approximation)

|   min |   p25 |   median |   mean |   std |   p75 |   p90 |   p95 |   p99 |   max |
|------:|------:|---------:|-------:|------:|------:|------:|------:|------:|------:|
|     5 |    94 |      168 |  218.8 | 211.2 |   280 |   426 |   557 |   986 |  5699 |

### Length buckets

| bucket                 |   rows |   pct |
|:-----------------------|-------:|------:|
| very short (<25 words) |   3411 |  3.16 |
| short (25-74)          |  17558 | 16.26 |
| medium (75-199)        |  41927 | 38.82 |
| long (200-499)         |  37876 | 35.07 |
| very long (500+)       |   7220 |  6.69 |

- Narratives under 10 words: 226
- Would exceed a 256-word window: 29.07%
- Would exceed a 512-word window: 6.24%

Those two figures drive the `max_length` fairness analysis required before the final model comparison (PROJECT_DIRECTIONS section 41). They are word counts, not sub-word tokens; DistilBERT's tokenizer will produce more tokens than words, so the real truncation rate at a 256-token limit is higher than shown here.

### Narrative length per Product

| product                                            |   median |   mean |   max |
|:---------------------------------------------------|---------:|-------:|------:|
| Checking or savings account                        |      189 |  237.5 |  5347 |
| Credit card                                        |      206 |  250.4 |  4950 |
| Debt collection                                    |      131 |  178.1 |  4080 |
| Money transfer, virtual currency, or money service |      163 |  211.5 |  5699 |
| Student loan                                       |      170 |  222.6 |  4994 |

### Duplicate narrative text

- Rows sharing exact text with another row: 7,035
- Distinct duplicated texts: 847
- Largest duplicate group: 832 rows
- Duplicate texts spanning more than one Product: 7

Duplicates are counted, not removed. They matter mainly as a train/test leakage risk, which is a Phase 4 split-design concern (PROJECT_DIRECTIONS section 27). A duplicate text appearing under more than one Product is a genuine label-consistency problem and is reported separately above.

### Vocabulary fingerprint (semantic separation proxy)

Words most over-represented in each Product relative to the rest of the corpus, from a per-product sample, excluding stopwords and the CFPB redaction token. A cheap check on whether the classes carry distinguishable vocabulary; it does not replace a trained model.

| product                                            | most_over_represented_words                                                                                                                                                                                |
|:---------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Checking or savings account                        | chexsystems (4.6x), poa (4.3x), federals (4.1x), axos (4.1x), atm (4.1x), regions (4.1x), hsa (4.1x), charles (4.0x), marcus (4.0x), overdrafts (3.9x), optum (3.9x), overdraft (3.8x)                     |
| Credit card                                        | dell (4.3x), sapphire (4.3x), jetblue (4.3x), concora (4.3x), avant (4.3x), nsettled (4.3x), tmerchant (4.3x), taxi (4.3x), mastercard (4.3x), barclay (4.2x), expresss (4.2x), comenity (4.2x)            |
| Debt collection                                    | cred (6.0x), exeter (6.0x), weiner (6.0x), sequium (6.0x), jeffcapsys (6.0x), coll (6.0x), tsi (6.0x), mcm (6.0x), hunter (6.0x), caine (6.0x), jefferson (6.0x), syst (6.0x)                              |
| Money transfer, virtual currency, or money service | moneygram (5.3x), unfeasible (5.3x), kraken (5.3x), coinme (5.3x), bitstamp (5.3x), abound (5.3x), offenders (5.3x), remitly (5.3x), zelles (5.2x), zelle (5.2x), eth (5.2x), western (5.2x)               |
| Student loan                                       | alumni (5.0x), dream (5.0x), unsubsidized (5.0x), navients (5.0x), ferpa (5.0x), servicers (5.0x), graduates (5.0x), educations (5.0x), mohelas (5.0x), buyback (5.0x), pslf (5.0x), capitalization (5.0x) |

Most frequent content words per Product:

- **Checking or savings account**: account (12,893), bank (6,902), funds (4,460), year (3,022), money (2,630), check (2,537), transactions (2,495), fraud (2,495), card (2,453), transaction (2,335), dispute (2,143), wells (2,046), credit (2,033), received (2,031), information (2,017), claim (1,991), unauthorized (1,982), fargo (1,888), complaint (1,793), time (1,766)
- **Credit card**: credit (9,097), account (9,072), card (7,370), bank (3,841), payment (3,770), dispute (3,702), year (2,824), balance (2,732), one (2,595), charges (2,493), interest (2,336), received (2,305), charge (2,279), information (2,260), merchant (2,105), time (2,097), complaint (1,905), made (1,893), never (1,860), capital (1,848)
- **Debt collection**: debt (10,856), account (9,086), credit (8,013), reporting (5,937), collection (4,987), information (3,616), validation (3,177), report (3,143), under (3,114), provide (2,641), documentation (2,632), original (2,516), request (2,370), act (2,365), fair (2,285), alleged (2,168), reported (2,060), dispute (2,029), consumer (1,982), balance (1,883)
- **Money transfer, virtual currency, or money service**: account (9,795), funds (4,356), money (3,724), bank (3,672), app (3,286), cash (3,167), paypal (3,035), transaction (2,687), year (2,554), transfer (2,208), received (1,960), sent (1,934), fraud (1,849), transactions (1,845), dispute (1,696), email (1,652), through (1,629), payment (1,526), back (1,520), financial (1,479)
- **Student loan**: loan (8,101), loans (5,722), mohela (4,415), payment (4,297), student (4,283), account (4,101), payments (3,765), credit (3,214), information (2,989), interest (2,843), due (2,459), forbearance (2,379), federal (2,362), time (2,164), year (2,063), received (2,032), financial (1,893), request (1,832), repayment (1,779), made (1,653)

---

## 8. Temporal audit

- Overall range: 2024-04-01 to 2026-06-30
- Distinct months: 27
- Unparseable dates: 0

| product                                            | earliest   | latest     |   count |
|:---------------------------------------------------|:-----------|:-----------|--------:|
| Checking or savings account                        | 2025-12-01 | 2026-06-30 |   21547 |
| Credit card                                        | 2025-12-01 | 2026-06-30 |   20890 |
| Debt collection                                    | 2026-03-01 | 2026-06-30 |   24007 |
| Money transfer, virtual currency, or money service | 2025-06-01 | 2026-06-30 |   21437 |
| Student loan                                       | 2024-04-01 | 2026-06-30 |   20111 |

### Monthly concentration

| product                                            |   months | peak_month   |   peak_month_pct |   expected_even_pct |
|:---------------------------------------------------|---------:|:-------------|-----------------:|--------------------:|
| Checking or savings account                        |        7 | 2026-01      |            17.6  |               14.29 |
| Credit card                                        |        7 | 2026-01      |            18.13 |               14.29 |
| Debt collection                                    |        4 | 2026-03      |            34.36 |               25    |
| Money transfer, virtual currency, or money service |       13 | 2025-10      |            12.29 |                7.69 |
| Student loan                                       |       27 | 2025-03      |             9.09 |                3.7  |

Per-product date ranges differ by design. The Phase 2B rule took the most recent months first until each product reached roughly 20,000 rows, so lower-volume products reach further back. Any temporal analysis or temporal split must account for this.
