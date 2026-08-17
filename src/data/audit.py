"""Phase 3 - raw CFPB data quality audit.

Measures the acquired raw extract so the final ML label set can be locked from
evidence rather than assumption: file/schema inventory, row-level integrity,
Product / Sub-product / Issue / Sub-issue structure, narrative quality,
temporal coverage, class imbalance, and a lightweight vocabulary proxy for
semantic separation between candidate classes.

This module does not clean, deduplicate, merge labels, balance classes, or
modify the raw data in any way. It reads and measures. Every derived artefact
is written under reports/, never under data/raw/.

The label-set recommendation is produced as evidence and a proposal for the
project owner to confirm; it is not applied automatically. See
PROJECT_DIRECTIONS.md sections 11, 23 and 71.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Fields PROJECT_DIRECTIONS section 17 requires the raw extract to retain.
# "Timely response?" and "ZIP code" carry the exact CFPB spellings.
EXPECTED_RAW_COLUMNS = [
    "Complaint ID",
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "ZIP code",
    "Submitted via",
    "Company response to consumer",
    "Timely response?",
    "Company public response",
    "Date sent to company",
]

PROVISIONAL_PRODUCTS = [
    "Debt collection",
    "Credit card",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
    "Student loan",
]

# CFPB masks personally identifiable text with this token before publication.
# It dominates raw word frequencies and carries no semantic content, so it is
# excluded from the vocabulary fingerprint.
REDACTION_TOKEN = "xxxx"

# A small stopword list is sufficient for a distinctiveness proxy. This is a
# diagnostic only - it is not the preprocessing used for modeling.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "is", "was", "were", "are", "be", "been", "being", "with", "as", "at",
    "by", "this", "that", "it", "my", "me", "have", "has", "had", "not",
    "no", "do", "does", "did", "from", "they", "them", "their", "you",
    "your", "we", "us", "our", "he", "she", "his", "her", "which", "will",
    "would", "could", "should", "can", "so", "if", "then", "than", "there",
    "these", "those", "am", "any", "all", "when", "what", "who", "how",
    "about", "into", "out", "up", "down", "over", "after", "before", "also",
    "because", "were", "said", "told", "get", "got", "will", "just", "only",
}

WORD_PATTERN = re.compile(r"[a-z']+")

# Word-count buckets for the narrative length profile. Chosen so the boundary
# between "medium" and "long" sits near the point where a 256-token
# transformer window starts truncating, which is the practical decision this
# profile has to inform.
LENGTH_BUCKETS = [
    ("very short (<25 words)", 0, 25),
    ("short (25-74)", 25, 75),
    ("medium (75-199)", 75, 200),
    ("long (200-499)", 200, 500),
    ("very long (500+)", 500, None),
]

# A Sub-product / Issue / Sub-issue category below this share of its parent is
# reported as a small category. It is a review marker, not a deletion rule.
SMALL_CATEGORY_PCT = 1.0


@dataclass(frozen=True)
class AuditConfig:
    """Resolved Phase 3 settings loaded from configs/data.yaml."""

    source_dir: str
    short_narrative_words: int
    top_n_issues: int
    top_n_subproducts: int
    vocabulary_sample_per_product: int
    vocabulary_top_n: int
    random_seed: int
    report_path: str
    label_decision_path: str


def load_config(path: Path) -> AuditConfig:
    """Load the Phase 3 audit configuration from a YAML file."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    audit = raw["audit"]
    return AuditConfig(
        source_dir=audit["source_dir"],
        short_narrative_words=int(audit["short_narrative_words"]),
        top_n_issues=int(audit["top_n_issues"]),
        top_n_subproducts=int(audit["top_n_subproducts"]),
        vocabulary_sample_per_product=int(audit["vocabulary_sample_per_product"]),
        vocabulary_top_n=int(audit["vocabulary_top_n"]),
        random_seed=int(audit["random_seed"]),
        report_path=audit["report_path"],
        label_decision_path=audit["label_decision_path"],
    )


# --------------------------------------------------------------------------
# 1. File and schema inventory
# --------------------------------------------------------------------------


def inventory_source_files(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect every raw CSV's schema without assuming filenames or columns.

    Returns a per-file inventory and a table of schema differences, so a
    column that is named differently in one file is surfaced rather than
    silently reconciled.
    """
    files = sorted(source_dir.rglob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found under {source_dir}")

    inventory_rows = []
    schema_by_file: dict[str, list[str]] = {}

    for path in files:
        header = pd.read_csv(path, nrows=0)
        columns = list(header.columns)
        schema_by_file[path.name] = columns

        # Narratives contain embedded newlines, so the row count has to come
        # from the parsed CSV rather than from counting physical lines.
        row_count = len(pd.read_csv(path, usecols=[columns[0]], dtype=str))
        inventory_rows.append(
            {
                "file": path.name,
                "rows": row_count,
                "columns": len(columns),
                "missing_expected_columns": ", ".join(
                    column for column in EXPECTED_RAW_COLUMNS if column not in columns
                )
                or "none",
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            }
        )

    reference_name, reference_columns = next(iter(schema_by_file.items()))
    difference_rows = []
    for name, columns in schema_by_file.items():
        if columns != reference_columns:
            difference_rows.append(
                {
                    "file": name,
                    "only_here": ", ".join(set(columns) - set(reference_columns))
                    or "-",
                    "missing_here": ", ".join(set(reference_columns) - set(columns))
                    or "-",
                    "compared_against": reference_name,
                }
            )

    return pd.DataFrame(inventory_rows), pd.DataFrame(difference_rows)


def load_raw_complaints(source_dir: Path) -> pd.DataFrame:
    """Load every raw CSV under the source directory into one audit view."""
    files = sorted(source_dir.rglob("*.csv"))
    frames = []
    for path in files:
        frame = pd.read_csv(path, dtype=str)
        frame["source_file"] = path.name
        frames.append(frame)
        logger.info("Loaded %s: %d rows", path.name, len(frame))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# 2. Combined dataset inventory
# --------------------------------------------------------------------------


def dataset_inventory(complaints: pd.DataFrame) -> dict[str, Any]:
    """Independently verify the integrity figures reported by Phase 2B."""
    total = len(complaints)
    complaint_ids = complaints["Complaint ID"]
    dates = pd.to_datetime(complaints["Date received"], errors="coerce")
    narrative = complaints["Consumer complaint narrative"]

    missingness = []
    for column in EXPECTED_RAW_COLUMNS:
        if column not in complaints.columns:
            missingness.append(
                {"field": column, "missing": total, "missing_pct": 100.0}
            )
            continue
        missing = int(complaints[column].isna().sum())
        missingness.append(
            {
                "field": column,
                "missing": missing,
                "missing_pct": round(100 * missing / total, 2) if total else 0.0,
            }
        )

    observed_products = sorted(complaints["Product"].dropna().unique())
    return {
        "total_rows": total,
        "unique_complaint_ids": int(complaint_ids.nunique()),
        "duplicate_complaint_ids": int(complaint_ids.duplicated().sum()),
        "missing_complaint_ids": int(complaint_ids.isna().sum()),
        "min_date_received": dates.min().date().isoformat(),
        "max_date_received": dates.max().date().isoformat(),
        "unparseable_dates": int(dates.isna().sum()),
        "missing_narrative": int(narrative.isna().sum()),
        "blank_narrative": int(
            (narrative.fillna("").str.strip() == "").sum() - narrative.isna().sum()
        ),
        "source_files": int(complaints["source_file"].nunique()),
        "rows_per_source_file": complaints["source_file"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "observed_products": observed_products,
        "unexpected_products": sorted(
            set(observed_products) - set(PROVISIONAL_PRODUCTS)
        ),
        "missing_expected_products": sorted(
            set(PROVISIONAL_PRODUCTS) - set(observed_products)
        ),
        "missingness": missingness,
    }


# --------------------------------------------------------------------------
# 3. Product label audit
# --------------------------------------------------------------------------


def product_audit(complaints: pd.DataFrame) -> pd.DataFrame:
    """Per-Product volume, share, and monthly volume profile."""
    dates = pd.to_datetime(complaints["Date received"], errors="coerce")
    working = complaints.assign(month=dates.dt.strftime("%Y-%m"))
    total = len(working)

    monthly = working.groupby(["Product", "month"]).size()

    rows = []
    for product, group in working.groupby("Product"):
        product_months = monthly.loc[product]
        rows.append(
            {
                "product": product,
                "count": len(group),
                "pct_of_dataset": round(100 * len(group) / total, 2),
                "months_observed": int(len(product_months)),
                "min_month": int(product_months.min()),
                "median_month": float(product_months.median()),
                "max_month": int(product_months.max()),
                "first_month": product_months.index.min(),
                "last_month": product_months.index.max(),
            }
        )

    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(
        drop=True
    )


def class_imbalance_summary(product_table: pd.DataFrame) -> dict[str, Any]:
    """Summarise imbalance between the largest and smallest candidate class."""
    largest = product_table.iloc[0]
    smallest = product_table.iloc[-1]
    return {
        "classes": int(len(product_table)),
        "largest_class": largest["product"],
        "largest_count": int(largest["count"]),
        "smallest_class": smallest["product"],
        "smallest_count": int(smallest["count"]),
        "imbalance_ratio": round(largest["count"] / smallest["count"], 2),
    }


# --------------------------------------------------------------------------
# 4-6. Sub-product / Issue / Sub-issue audits
# --------------------------------------------------------------------------


def categorical_within_product(
    complaints: pd.DataFrame, field: str, top_n: int
) -> pd.DataFrame:
    """Top values of a categorical field within each Product."""
    rows = []
    for product, group in complaints.groupby("Product"):
        counts = group[field].value_counts(dropna=False).head(top_n)
        total = len(group)
        for value, count in counts.items():
            rows.append(
                {
                    "product": product,
                    field: value if pd.notna(value) else "(missing)",
                    "count": int(count),
                    "pct_within_product": round(100 * count / total, 2),
                }
            )
    return pd.DataFrame(rows)


def categorical_structure(
    complaints: pd.DataFrame, field: str
) -> pd.DataFrame:
    """Per-Product structure of a categorical field.

    Reports how many distinct values each Product spans, how concentrated it
    is in its single largest value, and how many values are small enough to be
    review candidates. This is what determines whether a Product is internally
    homogeneous or heterogeneous.
    """
    rows = []
    for product, group in complaints.groupby("Product"):
        counts = group[field].value_counts(dropna=False)
        total = len(group)
        share = counts / total * 100
        rows.append(
            {
                "product": product,
                f"distinct_{field.lower().replace('-', '_')}": int(len(counts)),
                "top_value": (
                    str(counts.index[0]) if pd.notna(counts.index[0]) else "(missing)"
                ),
                "top_value_pct": round(share.iloc[0], 2),
                "top3_pct": round(share.iloc[:3].sum(), 2),
                f"categories_under_{int(SMALL_CATEGORY_PCT)}pct": int(
                    (share < SMALL_CATEGORY_PCT).sum()
                ),
                "missing_pct": round(
                    100 * group[field].isna().sum() / total, 2
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("product").reset_index(drop=True)


def categorical_overview(complaints: pd.DataFrame, field: str) -> dict[str, Any]:
    """Dataset-wide cardinality and concentration for a categorical field."""
    counts = complaints[field].value_counts(dropna=True)
    total = int(counts.sum())
    cumulative = counts.cumsum() / total * 100

    return {
        "distinct_values": int(len(counts)),
        "missing_rows": int(complaints[field].isna().sum()),
        "missing_pct": round(100 * complaints[field].isna().sum() / len(complaints), 2),
        "top10_share_pct": round(counts.head(10).sum() / total * 100, 2),
        "values_covering_80pct": int((cumulative <= 80).sum() + 1),
        "values_under_100_rows": int((counts < 100).sum()),
        "values_under_500_rows": int((counts < 500).sum()),
        "top_values": counts.head(15).to_dict(),
    }


def cross_product_value_overlap(
    complaints: pd.DataFrame, field: str
) -> pd.DataFrame:
    """Values of a categorical field that appear under more than one Product.

    The model classifies narrative text rather than this field, so overlap is
    supporting evidence about how distinct the Products are, not a verdict on
    its own.
    """
    value_products = (
        complaints.dropna(subset=[field])
        .groupby(field)["Product"]
        .agg(lambda values: sorted(set(values)))
    )
    shared = value_products[value_products.apply(len) > 1]
    if shared.empty:
        return pd.DataFrame()

    totals = complaints[field].value_counts()
    rows = [
        {
            field: value,
            "products": ", ".join(products),
            "product_count": len(products),
            "rows": int(totals.loc[value]),
        }
        for value, products in shared.items()
    ]
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(
        drop=True
    )


# --------------------------------------------------------------------------
# 7. Narrative quality audit
# --------------------------------------------------------------------------


def _describe_lengths(series: pd.Series) -> dict[str, Any]:
    """Standard descriptive statistics for a length series."""
    return {
        "min": int(series.min()),
        "p25": round(series.quantile(0.25), 1),
        "median": round(series.median(), 1),
        "mean": round(series.mean(), 1),
        "std": round(series.std(), 1),
        "p75": round(series.quantile(0.75), 1),
        "p90": round(series.quantile(0.90), 1),
        "p95": round(series.quantile(0.95), 1),
        "p99": round(series.quantile(0.99), 1),
        "max": int(series.max()),
    }


def narrative_audit(
    complaints: pd.DataFrame, short_narrative_words: int
) -> dict[str, Any]:
    """Character and word length profile of the raw narrative field.

    Word counts use a whitespace split. That is a reproducible exploratory
    approximation, deliberately not the tokenizer used for modeling - the
    LSTM and DistilBERT tokenizers will each produce different counts.
    """
    narrative = complaints["Consumer complaint narrative"]
    present = narrative.dropna()

    char_length = present.str.len()
    word_length = present.str.split().str.len()

    buckets = []
    for label, lower, upper in LENGTH_BUCKETS:
        if upper is None:
            mask = word_length >= lower
        else:
            mask = (word_length >= lower) & (word_length < upper)
        buckets.append(
            {
                "bucket": label,
                "rows": int(mask.sum()),
                "pct": round(100 * mask.sum() / len(word_length), 2),
            }
        )

    per_product = (
        complaints.assign(word_length=narrative.str.split().str.len())
        .groupby("Product")["word_length"]
        .agg(["median", "mean", "max"])
        .round(1)
        .reset_index()
        .rename(columns={"Product": "product"})
    )

    return {
        "rows_with_narrative": int(len(present)),
        "missing_narrative": int(narrative.isna().sum()),
        "blank_narrative": int((present.str.strip() == "").sum()),
        "char_length": _describe_lengths(char_length),
        "word_length": _describe_lengths(word_length),
        "buckets": buckets,
        "very_short_threshold": short_narrative_words,
        "very_short_count": int((word_length < short_narrative_words).sum()),
        "per_product": per_product,
        "truncation_at_256_words_pct": round(
            100 * (word_length > 256).sum() / len(word_length), 2
        ),
        "truncation_at_512_words_pct": round(
            100 * (word_length > 512).sum() / len(word_length), 2
        ),
    }


def narrative_duplicate_audit(complaints: pd.DataFrame) -> dict[str, Any]:
    """Exact-duplicate narrative text across distinct Complaint IDs.

    Complaint mills sometimes submit identical boilerplate under many
    Complaint IDs. Exact duplicates matter for leakage between train and test
    splits, so they are counted here and left for the Phase 4 split design.
    Nothing is removed.
    """
    narrative = complaints["Consumer complaint narrative"].dropna()
    duplicated_mask = narrative.duplicated(keep=False)
    groups = narrative[duplicated_mask].value_counts()

    cross_product = 0
    if not groups.empty:
        subset = complaints.loc[narrative[duplicated_mask].index]
        products_per_text = subset.groupby("Consumer complaint narrative")[
            "Product"
        ].nunique()
        cross_product = int((products_per_text > 1).sum())

    return {
        "rows_with_duplicate_text": int(duplicated_mask.sum()),
        "distinct_duplicate_groups": int(len(groups)),
        "largest_group": int(groups.iloc[0]) if not groups.empty else 0,
        "duplicate_groups_spanning_products": cross_product,
    }


def vocabulary_fingerprint(
    complaints: pd.DataFrame,
    sample_per_product: int,
    top_n: int,
    random_seed: int,
) -> tuple[dict[str, list[tuple[str, int]]], pd.DataFrame]:
    """Distinctive vocabulary per Product, as a separability proxy.

    Returns both the most frequent content words per product and, for each
    product, the words most over-represented relative to the rest of the
    corpus. The second view is the informative one: it shows whether a class
    has vocabulary a bag-of-words model could separate on.
    """
    samples: dict[str, Counter[str]] = {}
    for product, group in complaints.groupby("Product"):
        texts = group["Consumer complaint narrative"].dropna()
        if len(texts) > sample_per_product:
            texts = texts.sample(sample_per_product, random_state=random_seed)

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(
                word
                for word in WORD_PATTERN.findall(text.lower())
                if word not in STOPWORDS and word != REDACTION_TOKEN and len(word) > 2
            )
        samples[product] = counter

    frequent = {
        product: counter.most_common(top_n) for product, counter in samples.items()
    }

    corpus_total = sum(sum(counter.values()) for counter in samples.values())
    corpus_counts: Counter[str] = Counter()
    for counter in samples.values():
        corpus_counts.update(counter)

    distinctive_rows = []
    for product, counter in samples.items():
        product_total = sum(counter.values())
        scored = []
        for word, count in counter.items():
            if count < 30:
                continue
            product_rate = count / product_total
            corpus_rate = corpus_counts[word] / corpus_total
            scored.append((word, product_rate / corpus_rate, count))
        scored.sort(key=lambda item: -item[1])
        distinctive_rows.append(
            {
                "product": product,
                "most_over_represented_words": ", ".join(
                    f"{word} ({ratio:.1f}x)" for word, ratio, _ in scored[:12]
                ),
            }
        )

    return frequent, pd.DataFrame(distinctive_rows)


# --------------------------------------------------------------------------
# 8. Temporal audit
# --------------------------------------------------------------------------


def temporal_audit(complaints: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Date coverage overall, per product, and per month."""
    dates = pd.to_datetime(complaints["Date received"], errors="coerce")
    working = complaints.assign(month=dates.dt.strftime("%Y-%m"), parsed=dates)

    per_product = (
        working.groupby("Product")["parsed"]
        .agg(["min", "max", "count"])
        .reset_index()
        .rename(columns={"Product": "product", "min": "earliest", "max": "latest"})
    )
    per_product["earliest"] = per_product["earliest"].dt.date.astype(str)
    per_product["latest"] = per_product["latest"].dt.date.astype(str)

    monthly = (
        working.groupby(["Product", "month"]).size().rename("rows").reset_index()
    )

    # A month carrying an outsized share of its product signals concentration
    # that could bias a temporal split.
    concentration = []
    for product, group in monthly.groupby("Product"):
        share = group["rows"] / group["rows"].sum() * 100
        peak_index = share.idxmax()
        concentration.append(
            {
                "product": product,
                "months": int(len(group)),
                "peak_month": group.loc[peak_index, "month"],
                "peak_month_pct": round(share.max(), 2),
                "expected_even_pct": round(100 / len(group), 2),
            }
        )

    summary = {
        "overall_min": dates.min().date().isoformat(),
        "overall_max": dates.max().date().isoformat(),
        "unparseable_dates": int(dates.isna().sum()),
        "distinct_months": int(working["month"].nunique()),
        "per_product": per_product,
        "concentration": pd.DataFrame(concentration),
    }
    return summary, monthly


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def write_audit_report(path: Path, sections: dict[str, Any]) -> None:
    """Write the human-readable Phase 3 audit report."""
    inventory = sections["inventory"]
    products = sections["products"]
    imbalance = sections["imbalance"]
    narrative = sections["narrative"]
    duplicates = sections["duplicates"]
    temporal = sections["temporal"]

    lines = [
        "# Phase 3 - Raw Data Quality Audit",
        "",
        f"- Generated: {sections['generated_at']}",
        f"- Source: `{sections['source_dir']}`",
        f"- Rows audited: {inventory['total_rows']:,}",
        "",
        "This report measures the raw extract exactly as acquired. No cleaning, "
        "deduplication, label merging, balancing, or tokenization was performed, "
        "and no file under `data/raw/` was modified.",
        "",
        "---",
        "",
        "## 1. File and schema inventory",
        "",
        sections["file_inventory"].to_markdown(index=False),
        "",
    ]

    if sections["schema_differences"].empty:
        lines += [
            "All files share an identical schema, so the combined audit view "
            "needs no column reconciliation.",
            "",
        ]
    else:
        lines += [
            "**Schema differences detected between files:**",
            "",
            sections["schema_differences"].to_markdown(index=False),
            "",
        ]

    lines += [
        "---",
        "",
        "## 2. Combined dataset inventory",
        "",
        "Independently recomputed here rather than carried over from the "
        "Phase 2B report.",
        "",
        f"- Total rows: {inventory['total_rows']:,}",
        f"- Unique Complaint IDs: {inventory['unique_complaint_ids']:,}",
        f"- Duplicate Complaint IDs: {inventory['duplicate_complaint_ids']:,}",
        f"- Missing Complaint IDs: {inventory['missing_complaint_ids']:,}",
        f"- Date received range: {inventory['min_date_received']} to "
        f"{inventory['max_date_received']}",
        f"- Unparseable dates: {inventory['unparseable_dates']:,}",
        f"- Missing narratives: {inventory['missing_narrative']:,}",
        f"- Blank/whitespace-only narratives: {inventory['blank_narrative']:,}",
        f"- Source files: {inventory['source_files']}",
        f"- Unexpected Product values: "
        f"{inventory['unexpected_products'] or 'none'}",
        f"- Expected Products absent: "
        f"{inventory['missing_expected_products'] or 'none'}",
        "",
        "### Missing values by field",
        "",
        pd.DataFrame(inventory["missingness"]).to_markdown(index=False),
        "",
        "---",
        "",
        "## 3. Product label audit",
        "",
        products.to_markdown(index=False),
        "",
        f"- Candidate classes: {imbalance['classes']}",
        f"- Largest: {imbalance['largest_class']} "
        f"({imbalance['largest_count']:,})",
        f"- Smallest: {imbalance['smallest_class']} "
        f"({imbalance['smallest_count']:,})",
        f"- Imbalance ratio: **{imbalance['imbalance_ratio']}x**",
        "",
        "The near-even distribution is a direct consequence of the Phase 2B "
        "acquisition rule, which targeted roughly 20,000 rows per product. It "
        "is a property of this extract, not of the CFPB population, and the "
        "Phase 2A audit records the true population imbalance (Debt collection "
        "220,651 against Student loan 26,534, an 8.3x ratio).",
        "",
        "---",
        "",
        "## 4. Sub-product audit",
        "",
        "### Structure within each Product",
        "",
        sections["subproduct_structure"].to_markdown(index=False),
        "",
        "### Top Sub-products per Product",
        "",
        sections["subproducts"].to_markdown(index=False),
        "",
        "---",
        "",
        "## 5. Issue audit",
        "",
        f"- Distinct Issue values: {sections['issue_overview']['distinct_values']}",
        f"- Missing Issue: {sections['issue_overview']['missing_rows']:,} "
        f"({sections['issue_overview']['missing_pct']}%)",
        f"- Top 10 Issues cover "
        f"{sections['issue_overview']['top10_share_pct']}% of rows",
        f"- Issues needed to cover 80% of rows: "
        f"{sections['issue_overview']['values_covering_80pct']}",
        f"- Issues with under 500 rows: "
        f"{sections['issue_overview']['values_under_500_rows']}",
        "",
        "### Structure within each Product",
        "",
        sections["issue_structure"].to_markdown(index=False),
        "",
        "### Top Issues per Product",
        "",
        sections["issues"].to_markdown(index=False),
        "",
    ]

    if not sections["issue_overlap"].empty:
        lines += [
            "### Issues appearing under more than one Product",
            "",
            sections["issue_overlap"].head(15).to_markdown(index=False),
            "",
        ]
    else:
        lines += [
            "No Issue value appears under more than one Product: the Issue "
            "vocabulary is fully product-specific.",
            "",
        ]

    lines += [
        "---",
        "",
        "## 6. Sub-issue audit",
        "",
        f"- Distinct Sub-issue values: "
        f"{sections['subissue_overview']['distinct_values']}",
        f"- Missing Sub-issue: "
        f"{sections['subissue_overview']['missing_rows']:,} "
        f"({sections['subissue_overview']['missing_pct']}%)",
        f"- Sub-issues needed to cover 80% of rows: "
        f"{sections['subissue_overview']['values_covering_80pct']}",
        f"- Sub-issues with under 100 rows: "
        f"{sections['subissue_overview']['values_under_100_rows']}",
        f"- Sub-issues with under 500 rows: "
        f"{sections['subissue_overview']['values_under_500_rows']}",
        "",
        "### Structure within each Product",
        "",
        sections["subissue_structure"].to_markdown(index=False),
        "",
        "---",
        "",
        "## 7. Narrative quality audit",
        "",
        f"- Rows with a narrative: {narrative['rows_with_narrative']:,}",
        f"- Missing narrative: {narrative['missing_narrative']:,}",
        f"- Blank/whitespace-only: {narrative['blank_narrative']:,}",
        "",
        "### Character length",
        "",
        pd.DataFrame([narrative["char_length"]]).to_markdown(index=False),
        "",
        "### Word length (whitespace split; exploratory approximation)",
        "",
        pd.DataFrame([narrative["word_length"]]).to_markdown(index=False),
        "",
        "### Length buckets",
        "",
        pd.DataFrame(narrative["buckets"]).to_markdown(index=False),
        "",
        f"- Narratives under {narrative['very_short_threshold']} words: "
        f"{narrative['very_short_count']:,}",
        f"- Would exceed a 256-word window: "
        f"{narrative['truncation_at_256_words_pct']}%",
        f"- Would exceed a 512-word window: "
        f"{narrative['truncation_at_512_words_pct']}%",
        "",
        "Those two figures drive the `max_length` fairness analysis required "
        "before the final model comparison (PROJECT_DIRECTIONS section 41). "
        "They are word counts, not sub-word tokens; DistilBERT's tokenizer "
        "will produce more tokens than words, so the real truncation rate at a "
        "256-token limit is higher than shown here.",
        "",
        "### Narrative length per Product",
        "",
        narrative["per_product"].to_markdown(index=False),
        "",
        "### Duplicate narrative text",
        "",
        f"- Rows sharing exact text with another row: "
        f"{duplicates['rows_with_duplicate_text']:,}",
        f"- Distinct duplicated texts: "
        f"{duplicates['distinct_duplicate_groups']:,}",
        f"- Largest duplicate group: {duplicates['largest_group']:,} rows",
        f"- Duplicate texts spanning more than one Product: "
        f"{duplicates['duplicate_groups_spanning_products']:,}",
        "",
        "Duplicates are counted, not removed. They matter mainly as a "
        "train/test leakage risk, which is a Phase 4 split-design concern "
        "(PROJECT_DIRECTIONS section 27). A duplicate text appearing under "
        "more than one Product is a genuine label-consistency problem and is "
        "reported separately above.",
        "",
        "### Vocabulary fingerprint (semantic separation proxy)",
        "",
        "Words most over-represented in each Product relative to the rest of "
        "the corpus, from a per-product sample, excluding stopwords and the "
        "CFPB redaction token. A cheap check on whether the classes carry "
        "distinguishable vocabulary; it does not replace a trained model.",
        "",
        sections["distinctive_vocabulary"].to_markdown(index=False),
        "",
        "Most frequent content words per Product:",
        "",
    ]

    for product, words in sections["frequent_vocabulary"].items():
        top = ", ".join(f"{word} ({count:,})" for word, count in words)
        lines.append(f"- **{product}**: {top}")

    lines += [
        "",
        "---",
        "",
        "## 8. Temporal audit",
        "",
        f"- Overall range: {temporal['overall_min']} to {temporal['overall_max']}",
        f"- Distinct months: {temporal['distinct_months']}",
        f"- Unparseable dates: {temporal['unparseable_dates']:,}",
        "",
        temporal["per_product"].to_markdown(index=False),
        "",
        "### Monthly concentration",
        "",
        temporal["concentration"].to_markdown(index=False),
        "",
        "Per-product date ranges differ by design. The Phase 2B rule took the "
        "most recent months first until each product reached roughly 20,000 "
        "rows, so lower-volume products reach further back. Any temporal "
        "analysis or temporal split must account for this.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Execute the Phase 3 raw data audit end to end."""
    config = load_config(config_path)
    source_dir = project_root / config.source_dir

    file_inventory, schema_differences = inventory_source_files(source_dir)
    complaints = load_raw_complaints(source_dir)
    logger.info("Auditing %d raw rows", len(complaints))

    inventory = dataset_inventory(complaints)
    products = product_audit(complaints)
    imbalance = class_imbalance_summary(products)

    subproducts = categorical_within_product(
        complaints, "Sub-product", config.top_n_subproducts
    )
    subproduct_structure = categorical_structure(complaints, "Sub-product")
    issues = categorical_within_product(complaints, "Issue", config.top_n_issues)
    issue_structure = categorical_structure(complaints, "Issue")
    issue_overview = categorical_overview(complaints, "Issue")
    issue_overlap = cross_product_value_overlap(complaints, "Issue")
    subissue_structure = categorical_structure(complaints, "Sub-issue")
    subissue_overview = categorical_overview(complaints, "Sub-issue")

    narrative = narrative_audit(complaints, config.short_narrative_words)
    duplicates = narrative_duplicate_audit(complaints)
    frequent_vocabulary, distinctive_vocabulary = vocabulary_fingerprint(
        complaints,
        config.vocabulary_sample_per_product,
        config.vocabulary_top_n,
        config.random_seed,
    )
    temporal, monthly = temporal_audit(complaints)

    sections = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_dir": config.source_dir,
        "file_inventory": file_inventory,
        "schema_differences": schema_differences,
        "inventory": inventory,
        "products": products,
        "imbalance": imbalance,
        "subproducts": subproducts,
        "subproduct_structure": subproduct_structure,
        "issues": issues,
        "issue_structure": issue_structure,
        "issue_overview": issue_overview,
        "issue_overlap": issue_overlap,
        "subissue_structure": subissue_structure,
        "subissue_overview": subissue_overview,
        "narrative": narrative,
        "duplicates": duplicates,
        "frequent_vocabulary": frequent_vocabulary,
        "distinctive_vocabulary": distinctive_vocabulary,
        "temporal": temporal,
        "monthly": monthly,
    }

    report_path = project_root / config.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_audit_report(report_path, sections)
    logger.info("Saved audit report: %s", report_path)

    monthly_path = report_path.with_name("phase3_monthly_counts.csv")
    monthly.to_csv(monthly_path, index=False)
    logger.info("Saved monthly counts: %s", monthly_path)

    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 3 raw data audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the data configuration.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    project_root = Path(__file__).resolve().parents[2]
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path

    run_audit(config_path, project_root)


if __name__ == "__main__":
    main()
