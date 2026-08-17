"""Phase 2A - CFPB complaint count audit.

Retrieves monthly complaint counts per CFPB Product from the Consumer Complaint
Database Trends endpoint, so that acquisition windows for Phase 2B can be chosen
from measured volume rather than from assumed monthly averages.

This module deliberately does not download, clean, or tokenize any complaint
narrative text. It only counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

# The Trends endpoint rejects lens=product unless a focus or sub_lens is also
# supplied ("Either Focus or Sub-lens is required for lens 'product'"), so a
# sub_lens is sent even though Phase 2A only consumes the product-level counts.
REQUIRED_SUB_LENS = "sub_product"

# trend_depth must be at least 5 per the API spec; it also has to be large
# enough that every requested product survives as its own bucket.
MIN_TREND_DEPTH = 5


@dataclass(frozen=True)
class AuditConfig:
    """Resolved Phase 2A settings loaded from configs/data.yaml."""

    base_url: str
    trends_endpoint: str
    start_date: str
    end_date: str
    products: list[str]
    has_narrative: str
    sparse_product_month: int
    oversized_product_month: int
    very_sparse_product_total: int
    timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: int

    @property
    def trends_url(self) -> str:
        return self.base_url.rstrip("/") + "/" + self.trends_endpoint.strip("/")


def load_config(path: Path) -> AuditConfig:
    """Load the Phase 2A audit configuration from a YAML file."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    thresholds = raw["thresholds"]
    http = raw["http"]
    return AuditConfig(
        base_url=raw["source"]["base_url"],
        trends_endpoint=raw["source"]["trends_endpoint"],
        start_date=raw["date_range"]["start"],
        end_date=raw["date_range"]["end"],
        products=list(raw["products"]),
        has_narrative=str(raw["has_narrative"]),
        sparse_product_month=thresholds["sparse_product_month"],
        oversized_product_month=thresholds["oversized_product_month"],
        very_sparse_product_total=thresholds["very_sparse_product_total"],
        timeout_seconds=http["timeout_seconds"],
        max_retries=http["max_retries"],
        retry_backoff_seconds=http["retry_backoff_seconds"],
    )


def build_trends_params(config: AuditConfig) -> list[tuple[str, str]]:
    """Build the query parameters for a single product-by-month trends request."""
    params: list[tuple[str, str]] = [
        ("lens", "product"),
        ("sub_lens", REQUIRED_SUB_LENS),
        ("trend_interval", "month"),
        ("trend_depth", str(max(MIN_TREND_DEPTH, len(config.products)))),
        ("date_received_min", config.start_date),
        ("date_received_max", config.end_date),
        ("has_narrative", config.has_narrative),
    ]
    params.extend(("product", product) for product in config.products)
    return params


def fetch_trends(config: AuditConfig) -> dict[str, Any]:
    """Call the CFPB Trends endpoint once and return the parsed JSON response.

    The endpoint intermittently returns HTTP 424 ("error calling OpenSearch")
    under load, so transient failures are retried with a fixed backoff.
    """
    params = build_trends_params(config)
    last_status: int | None = None

    for attempt in range(1, config.max_retries + 1):
        try:
            response = requests.get(
                config.trends_url,
                params=params,
                timeout=config.timeout_seconds,
                headers={"Accept": "application/json"},
            )
        except requests.RequestException as exc:
            logger.error("CFPB trends request failed on attempt %d: %s", attempt, exc)
            if attempt == config.max_retries:
                raise
            time.sleep(config.retry_backoff_seconds)
            continue

        last_status = response.status_code
        if response.status_code == 200:
            logger.info("Trends request succeeded on attempt %d", attempt)
            return response.json()

        logger.warning(
            "Trends request returned HTTP %d on attempt %d: %s",
            response.status_code,
            attempt,
            response.text[:200],
        )
        if attempt < config.max_retries:
            time.sleep(config.retry_backoff_seconds)

    raise RuntimeError(
        f"CFPB trends request failed after {config.max_retries} attempts "
        f"(last status {last_status})"
    )


def parse_product_month_counts(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten the trends response into one row per product-month.

    The counts live at aggregations.product.product.buckets[], where each
    product bucket carries its own trend_period histogram.
    """
    try:
        product_buckets = payload["aggregations"]["product"]["product"]["buckets"]
    except KeyError as exc:
        raise ValueError(f"Unexpected trends response shape: missing {exc}") from exc

    rows = []
    for bucket in product_buckets:
        product = bucket["key"]
        for period in bucket.get("trend_period", {}).get("buckets", []):
            month = datetime.fromisoformat(
                period["key_as_string"].replace("Z", "+00:00")
            ).date()
            rows.append(
                {
                    "product": product,
                    "month": month.isoformat()[:7],
                    "complaint_count": period["doc_count"],
                }
            )

    if not rows:
        raise ValueError("Trends response contained no product-month buckets")

    counts = pd.DataFrame(rows)
    return counts.sort_values(["product", "month"]).reset_index(drop=True)


def flag_product_months(counts: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Attach the project review flags to each product-month row."""
    flagged = counts.copy()
    flagged["is_sparse"] = flagged["complaint_count"] < config.sparse_product_month
    flagged["is_oversized"] = (
        flagged["complaint_count"] > config.oversized_product_month
    )
    return flagged


def summarise_products(counts: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Summarise total and typical monthly volume for each product."""
    summary = (
        counts.groupby("product")["complaint_count"]
        .agg(
            total_complaints="sum",
            months_observed="count",
            min_month_count="min",
            median_month_count="median",
            max_month_count="max",
        )
        .reset_index()
    )
    summary["is_very_sparse_overall"] = (
        summary["total_complaints"] < config.very_sparse_product_total
    )
    return summary.sort_values("total_complaints", ascending=False).reset_index(
        drop=True
    )


def recommend_windows(summary: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Derive a per-product acquisition window from observed monthly volume.

    The window size is chosen so a single request stays under the oversized
    threshold, using the product's peak observed month rather than an assumed
    monthly average.
    """
    recommendations = []
    for row in summary.itertuples():
        peak = row.max_month_count
        if peak > config.oversized_product_month:
            window = "split-month"
            months_per_request = 0
        else:
            # How many peak-sized months fit under the per-request budget,
            # capped at the span actually covered by the source population so
            # the recommendation cannot exceed the data that exists.
            fits = max(1, int(config.oversized_product_month // max(peak, 1)))
            months_per_request = min(fits, row.months_observed)
            window = f"{months_per_request}-month"

        recommendations.append(
            {
                "product": row.product,
                "peak_month_count": peak,
                "median_month_count": row.median_month_count,
                "recommended_window": window,
                "months_per_request": months_per_request,
                "estimated_requests": (
                    "n/a"
                    if months_per_request == 0
                    else -(-row.months_observed // months_per_request)
                ),
            }
        )
    return pd.DataFrame(recommendations)


def detect_incomplete_tail_months(
    counts: pd.DataFrame, drop_ratio: float = 0.5
) -> list[str]:
    """Identify trailing months whose volume collapses across every product.

    The CFPB only publishes a narrative once the company has responded, so the
    most recent months are still filling in. A month where every product drops
    below `drop_ratio` of its own median is treated as not yet complete rather
    than as genuinely low demand.
    """
    medians = counts.groupby("product")["complaint_count"].median()
    monthly = counts.pivot(
        index="month", columns="product", values="complaint_count"
    ).sort_index()

    incomplete = []
    for month in reversed(monthly.index.tolist()):
        row = monthly.loc[month]
        if ((row < medians * drop_ratio).all()) and not row.isna().any():
            incomplete.append(month)
        else:
            break
    return sorted(incomplete)


def detect_volume_anomalies(
    counts: pd.DataFrame, spike_multiple: float = 5.0
) -> pd.DataFrame:
    """Flag product-months whose volume far exceeds that product's median.

    A single extreme month can dominate a class and distort macro-F1, so these
    are surfaced for review before any sampling decision is made.
    """
    medians = counts.groupby("product")["complaint_count"].transform("median")
    anomalies = counts[counts["complaint_count"] > medians * spike_multiple].copy()
    anomalies["product_median"] = medians[anomalies.index]
    anomalies["times_median"] = (
        anomalies["complaint_count"] / anomalies["product_median"]
    ).round(1)
    return anomalies[
        ["product", "month", "complaint_count", "product_median", "times_median"]
    ].sort_values("times_median", ascending=False)


def write_audit_report(
    path: Path,
    config: AuditConfig,
    counts: pd.DataFrame,
    summary: pd.DataFrame,
    windows: pd.DataFrame,
    retrieved_at: str,
) -> None:
    """Write the human-readable Phase 2A audit report."""
    sparse = counts[counts["is_sparse"]]
    oversized = counts[counts["is_oversized"]]

    lines = [
        "# Phase 2A - CFPB Count Audit",
        "",
        f"- Retrieved: {retrieved_at}",
        f"- Source population: {config.start_date} to {config.end_date}",
        f"- Narrative filter: has_narrative={config.has_narrative}",
        f"- Products audited: {len(config.products)}",
        f"- Product-months observed: {len(counts)}",
        "",
        "## Per-product totals (narrative-bearing complaints)",
        "",
        summary.to_markdown(index=False),
        "",
        "## Derived acquisition windows",
        "",
        "Window sizes are computed from each product's peak observed month against "
        f"the oversized threshold of {config.oversized_product_month:,} "
        "complaints per request, rather than from an assumed monthly average.",
        "",
        windows.to_markdown(index=False),
        "",
        "## Review flags",
        "",
        f"- Sparse product-months (< {config.sparse_product_month:,}): {len(sparse)}",
        f"- Oversized product-months (> {config.oversized_product_month:,}): {len(oversized)}",
        f"- Very sparse products (< {config.very_sparse_product_total:,} total): "
        f"{int(summary['is_very_sparse_overall'].sum())}",
        "",
        "Flags are review markers. No product or month is removed on the basis of "
        "a flag; the label set is decided in Phase 3 after the raw data audit.",
        "",
    ]

    incomplete = detect_incomplete_tail_months(counts)
    if incomplete:
        lines += [
            "## Incomplete trailing months",
            "",
            "Volume collapses across every product in the following month(s), which "
            "indicates the CFPB publication lag rather than a genuine drop in "
            "complaints. A narrative is only published after the company responds, "
            "so the most recent months are still filling in.",
            "",
            *(f"- {month}" for month in incomplete),
            "",
            f"Recommendation: treat the source population as ending before "
            f"{incomplete[0]} for Phase 2B acquisition, and record the effective "
            "end date in the acquisition manifest.",
            "",
        ]

    anomalies = detect_volume_anomalies(counts)
    if not anomalies.empty:
        lines += [
            "## Volume anomalies",
            "",
            "Product-months exceeding five times that product's median volume. A "
            "single dominant month can skew the class distribution and therefore "
            "macro-F1, so these require a sampling decision in Phase 4.",
            "",
            anomalies.to_markdown(index=False),
            "",
        ]

    if not sparse.empty:
        lines += [
            "### Sparse product-months",
            "",
            sparse[["product", "month", "complaint_count"]].to_markdown(index=False),
            "",
        ]

    if not oversized.empty:
        lines += [
            "### Oversized product-months",
            "",
            oversized[["product", "month", "complaint_count"]].to_markdown(index=False),
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(config_path: Path, project_root: Path) -> None:
    """Execute the Phase 2A count audit end to end."""
    config = load_config(config_path)
    logger.info(
        "Auditing %d products from %s to %s",
        len(config.products),
        config.start_date,
        config.end_date,
    )

    payload = fetch_trends(config)
    retrieved_at = datetime.now().astimezone().isoformat(timespec="seconds")

    counts = flag_product_months(parse_product_month_counts(payload), config)
    summary = summarise_products(counts, config)
    windows = recommend_windows(summary, config)

    raw_path = project_root / "reports" / "phase2a_trends_raw.json"
    counts_path = project_root / "reports" / "phase2a_product_month_counts.csv"
    report_path = project_root / "reports" / "phase2a_count_audit.md"

    raw_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    counts.to_csv(counts_path, index=False)
    write_audit_report(report_path, config, counts, summary, windows, retrieved_at)

    logger.info("Product-months parsed: %d", len(counts))
    logger.info("Saved raw response: %s", raw_path)
    logger.info("Saved counts table: %s", counts_path)
    logger.info("Saved audit report: %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 2A CFPB count audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the data acquisition config.",
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
