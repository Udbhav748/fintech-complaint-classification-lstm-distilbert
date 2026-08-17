"""Phase 2B - CFPB raw narrative acquisition.

Downloads narrative-bearing complaints for the provisional product scope using
date-windowed CSV requests, one immutable raw file per product-month, and
records an audit trail in the acquisition manifest.

Window selection is driven by the Phase 2A monthly counts rather than by an
assumed monthly average: acquisition walks backwards from the effective end
month and stops once a product has accumulated its row target. That keeps the
extract recent, keeps the windows non-overlapping, and avoids downloading the
full 552,157-row source population.

This module deliberately performs no cleaning, deduplication, label merging,
tokenization, or balancing. Raw means raw. It only measures what it retrieved.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import time
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

# The search endpoint returns aggregations alongside the hits by default. They
# are already known from Phase 2A and only inflate the response, so they are
# switched off.
SEARCH_PARAMS_BASE = [("format", "csv"), ("no_aggs", "true")]

# Requested row cap per window. The largest product-month in the Phase 2A
# counts is 41,618, so this sits comfortably above any single window while
# still being low enough that a truncated response is detectable.
WINDOW_SIZE_CAP = 60_000

# Raw CSV columns that must survive acquisition, per PROJECT_DIRECTIONS
# section 17. Their absence is recorded rather than silently tolerated.
REQUIRED_RAW_COLUMNS = [
    "Complaint ID",
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Submitted via",
    "Company response to consumer",
    "Timely response?",
    "Company public response",
    "Date sent to company",
]


@dataclass(frozen=True)
class AcquisitionConfig:
    """Resolved Phase 2B settings loaded from configs/data.yaml."""

    base_url: str
    products: list[str]
    has_narrative: str
    effective_end_month: str
    target_rows_per_product: int
    request_delay_seconds: float
    source_dir: str
    manifest_path: str
    integrity_report: str
    counts_csv: str
    timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: int

    @property
    def search_url(self) -> str:
        return self.base_url


def load_config(path: Path) -> AcquisitionConfig:
    """Load the Phase 2B acquisition configuration from a YAML file."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    acquisition = raw["acquisition"]
    http = raw["http"]
    return AcquisitionConfig(
        base_url=raw["source"]["base_url"],
        products=list(raw["products"]),
        has_narrative=str(raw["has_narrative"]),
        effective_end_month=str(acquisition["effective_end_month"]),
        target_rows_per_product=int(acquisition["target_rows_per_product"]),
        request_delay_seconds=float(acquisition["request_delay_seconds"]),
        source_dir=acquisition["source_dir"],
        manifest_path=acquisition["manifest_path"],
        integrity_report=acquisition["integrity_report"],
        counts_csv=raw["output"]["counts_csv"],
        timeout_seconds=http["timeout_seconds"],
        max_retries=http["max_retries"],
        retry_backoff_seconds=http["retry_backoff_seconds"],
    )


def product_slug(product: str) -> str:
    """Return a filesystem-safe identifier for a CFPB product name."""
    return re.sub(r"[^a-z0-9]+", "_", product.lower()).strip("_")


def month_bounds(month: str) -> tuple[date, date]:
    """Return the first and last calendar day of a YYYY-MM month."""
    year, month_number = (int(part) for part in month.split("-"))
    last_day = monthrange(year, month_number)[1]
    return date(year, month_number, 1), date(year, month_number, last_day)


def plan_windows(counts: pd.DataFrame, config: AcquisitionConfig) -> pd.DataFrame:
    """Choose the acquisition windows for every product from the Phase 2A counts.

    For each product the most recent complete month is taken first and earlier
    months are added until the cumulative Phase 2A count reaches the row target.
    The month that crosses the target is included, so the plan slightly exceeds
    the target rather than falling short of it.
    """
    eligible = counts[counts["month"] <= config.effective_end_month]

    planned = []
    for product in config.products:
        product_months = eligible[eligible["product"] == product].sort_values(
            "month", ascending=False
        )
        if product_months.empty:
            raise ValueError(f"No Phase 2A counts available for product {product!r}")

        cumulative = 0
        for row in product_months.itertuples():
            window_start, window_end = month_bounds(row.month)
            cumulative += row.complaint_count
            planned.append(
                {
                    "product": product,
                    "month": row.month,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "expected_rows": int(row.complaint_count),
                    "cumulative_expected_rows": cumulative,
                }
            )
            if cumulative >= config.target_rows_per_product:
                break

        logger.info(
            "Planned %s: %d monthly windows, %s expected rows",
            product,
            sum(1 for entry in planned if entry["product"] == product),
            f"{cumulative:,}",
        )

    plan = pd.DataFrame(planned)
    return plan.sort_values(["product", "month"]).reset_index(drop=True)


def build_search_params(
    product: str, window_start: str, window_end: str, config: AcquisitionConfig
) -> list[tuple[str, str]]:
    """Build the query parameters for one product-month CSV request."""
    return [
        ("product", product),
        ("has_narrative", config.has_narrative),
        ("date_received_min", window_start),
        ("date_received_max", window_end),
        ("size", str(WINDOW_SIZE_CAP)),
        *SEARCH_PARAMS_BASE,
    ]


@dataclass
class WindowResult:
    """Outcome of a single product-month request."""

    status: str
    http_status: int | None = None
    attempts: int = 0
    error: str | None = None
    frame: pd.DataFrame | None = field(default=None, repr=False)
    raw_text: str | None = field(default=None, repr=False)


def fetch_window(
    product: str, window_start: str, window_end: str, config: AcquisitionConfig
) -> WindowResult:
    """Retrieve one product-month as CSV, retrying transient failures.

    The CFPB edge answers with HTTP 403 when it rate-limits a client and with
    HTTP 424 when the OpenSearch backend errors, so both are retried with a
    fixed backoff before the window is recorded as failed.
    """
    params = build_search_params(product, window_start, window_end, config)
    last_status: int | None = None
    last_error: str | None = None

    for attempt in range(1, config.max_retries + 1):
        try:
            response = requests.get(
                config.search_url,
                params=params,
                timeout=config.timeout_seconds,
                headers={"Accept": "text/csv"},
            )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "%s %s request failed on attempt %d: %s",
                product,
                window_start,
                attempt,
                exc,
            )
            if attempt < config.max_retries:
                time.sleep(config.retry_backoff_seconds * attempt)
            continue

        last_status = response.status_code
        if response.status_code == 200:
            frame = pd.read_csv(io.StringIO(response.text), dtype=str)
            return WindowResult(
                status="success",
                http_status=200,
                attempts=attempt,
                frame=frame,
                raw_text=response.text,
            )

        last_error = response.text[:200]
        logger.warning(
            "%s %s returned HTTP %d on attempt %d",
            product,
            window_start,
            response.status_code,
            attempt,
        )
        if attempt < config.max_retries:
            time.sleep(config.retry_backoff_seconds * attempt)

    return WindowResult(
        status="failed",
        http_status=last_status,
        attempts=config.max_retries,
        error=last_error,
    )


def file_checksum(path: Path) -> str:
    """Return the SHA-256 checksum of a raw file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarise_window(frame: pd.DataFrame) -> dict[str, Any]:
    """Measure what a retrieved window actually contains.

    These are observations only. Nothing is corrected, dropped, or normalised
    here; problems are recorded so Phase 3 can act on them.
    """
    dates = pd.to_datetime(frame.get("Date received"), errors="coerce")
    narrative = frame.get("Consumer complaint narrative")
    complaint_ids = frame.get("Complaint ID")

    return {
        "retrieved_rows": int(len(frame)),
        "min_date_received": None if dates.isna().all() else dates.min().date().isoformat(),
        "max_date_received": None if dates.isna().all() else dates.max().date().isoformat(),
        "missing_narrative": int(narrative.isna().sum()) if narrative is not None else None,
        "duplicate_complaint_ids": (
            int(complaint_ids.duplicated().sum()) if complaint_ids is not None else None
        ),
        "missing_required_columns": [
            column for column in REQUIRED_RAW_COLUMNS if column not in frame.columns
        ],
        "response_truncated": bool(len(frame) >= WINDOW_SIZE_CAP),
    }


def acquire_windows(
    plan: pd.DataFrame, config: AcquisitionConfig, project_root: Path
) -> list[dict[str, Any]]:
    """Retrieve every planned window and write it as an immutable raw file."""
    source_dir = project_root / config.source_dir
    entries: list[dict[str, Any]] = []

    for position, row in enumerate(plan.itertuples(), start=1):
        logger.info(
            "[%d/%d] %s  %s -> %s  (expected %s rows)",
            position,
            len(plan),
            row.product,
            row.window_start,
            row.window_end,
            f"{row.expected_rows:,}",
        )

        result = fetch_window(row.product, row.window_start, row.window_end, config)
        slug = product_slug(row.product)
        relative_path = f"{config.source_dir}/{slug}/{slug}_{row.month}.csv"

        entry: dict[str, Any] = {
            "product": row.product,
            "month": row.month,
            "window_start": row.window_start,
            "window_end": row.window_end,
            "expected_rows_phase2a": int(row.expected_rows),
            "request_status": result.status,
            "http_status": result.http_status,
            "attempts": result.attempts,
            "retries": max(0, result.attempts - 1),
            "error": result.error,
            "file_path": relative_path,
        }

        if result.status == "success" and result.frame is not None:
            product_dir = source_dir / slug
            product_dir.mkdir(parents=True, exist_ok=True)
            output_path = product_dir / f"{slug}_{row.month}.csv"
            output_path.write_text(result.raw_text or "", encoding="utf-8", newline="")

            measurements = summarise_window(result.frame)
            entry.update(measurements)
            entry["row_delta_vs_phase2a"] = (
                measurements["retrieved_rows"] - int(row.expected_rows)
            )
            entry["file_size_bytes"] = output_path.stat().st_size
            entry["sha256"] = file_checksum(output_path)

            logger.info(
                "    retrieved %s rows (delta %+d), saved %s",
                f"{measurements['retrieved_rows']:,}",
                entry["row_delta_vs_phase2a"],
                relative_path,
            )
        else:
            entry["file_path"] = None
            logger.error(
                "    FAILED after %d attempts (last HTTP %s)",
                result.attempts,
                result.http_status,
            )

        entries.append(entry)
        time.sleep(config.request_delay_seconds)

    return entries


def load_acquired_frame(entries: list[dict[str, Any]], project_root: Path) -> pd.DataFrame:
    """Concatenate every successfully saved raw file for the integrity checks."""
    frames = []
    for entry in entries:
        if entry["request_status"] != "success" or not entry["file_path"]:
            continue
        frames.append(pd.read_csv(project_root / entry["file_path"], dtype=str))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_integrity_checks(complaints: pd.DataFrame) -> dict[str, Any]:
    """Run the basic Phase 2B integrity checks over the acquired raw data.

    Deliberately limited to counting: the full data-quality audit and the label
    decision belong to Phase 3.
    """
    if complaints.empty:
        return {"total_rows": 0}

    dates = pd.to_datetime(complaints["Date received"], errors="coerce")
    return {
        "total_rows": int(len(complaints)),
        "rows_per_product": complaints["Product"].value_counts().to_dict(),
        "min_date_received": dates.min().date().isoformat(),
        "max_date_received": dates.max().date().isoformat(),
        "missing_narrative": int(
            complaints["Consumer complaint narrative"].isna().sum()
        ),
        "duplicate_complaint_ids": int(complaints["Complaint ID"].duplicated().sum()),
        "unique_complaint_ids": int(complaints["Complaint ID"].nunique()),
    }


def build_manifest(
    config: AcquisitionConfig,
    plan: pd.DataFrame,
    entries: list[dict[str, Any]],
    integrity: dict[str, Any],
    retrieved_at: str,
) -> dict[str, Any]:
    """Assemble the Phase 2B acquisition manifest."""
    successes = [entry for entry in entries if entry["request_status"] == "success"]
    failures = [entry for entry in entries if entry["request_status"] != "success"]

    return {
        "phase": "2B - Raw Narrative Acquisition",
        "source": "CFPB Consumer Complaint Database",
        "source_url": "https://www.consumerfinance.gov/data-research/consumer-complaints/",
        "endpoint": config.search_url,
        "retrieved_at": retrieved_at,
        "source_period": {
            "nominal_start": "2023-08-17",
            "nominal_end": "2026-08-17",
            "effective_end_month": config.effective_end_month,
            "effective_end_reason": (
                "2026-07 is incomplete because the CFPB publishes a narrative only "
                "after the company responds (Phase 2A finding)."
            ),
        },
        "filters": {
            "products": config.products,
            "has_narrative": config.has_narrative,
            "window": "calendar month, non-overlapping",
        },
        "query_parameters": {
            "format": "csv",
            "no_aggs": "true",
            "size": WINDOW_SIZE_CAP,
            "date_received_min": "window start (inclusive)",
            "date_received_max": "window end",
        },
        "selection_rule": (
            "Per product, months are taken from the most recent complete month "
            "backwards until the cumulative Phase 2A count reaches "
            f"{config.target_rows_per_product:,} rows."
        ),
        "planned_windows": int(len(plan)),
        "successful_windows": len(successes),
        "failed_windows": len(failures),
        "total_retries": sum(entry.get("retries", 0) for entry in entries),
        "integrity": integrity,
        "windows": entries,
    }


def write_integrity_report(
    path: Path,
    config: AcquisitionConfig,
    plan: pd.DataFrame,
    entries: list[dict[str, Any]],
    integrity: dict[str, Any],
    retrieved_at: str,
) -> None:
    """Write the human-readable Phase 2B acquisition report."""
    successes = [entry for entry in entries if entry["request_status"] == "success"]
    failures = [entry for entry in entries if entry["request_status"] != "success"]

    coverage = pd.DataFrame(
        [
            {
                "product": entry["product"],
                "month": entry["month"],
                "expected_rows_phase2a": entry["expected_rows_phase2a"],
                "retrieved_rows": entry.get("retrieved_rows"),
                "row_delta": entry.get("row_delta_vs_phase2a"),
                "status": entry["request_status"],
            }
            for entry in entries
        ]
    )

    lines = [
        "# Phase 2B - Raw Narrative Acquisition",
        "",
        f"- Retrieved: {retrieved_at}",
        f"- Effective end month: {config.effective_end_month}",
        f"- Per-product row target: {config.target_rows_per_product:,}",
        f"- Planned windows: {len(plan)}",
        f"- Successful windows: {len(successes)}",
        f"- Failed windows: {len(failures)}",
        "",
        "## Integrity checks",
        "",
        f"- Total rows: {integrity.get('total_rows', 0):,}",
    ]

    if integrity.get("total_rows"):
        rows_per_product = pd.Series(
            integrity["rows_per_product"], name="rows"
        ).sort_values(ascending=False)
        rows_per_product.index.name = "product"
        lines += [
            f"- Date received range: {integrity['min_date_received']} to "
            f"{integrity['max_date_received']}",
            f"- Missing narratives: {integrity['missing_narrative']:,}",
            f"- Duplicate Complaint IDs: {integrity['duplicate_complaint_ids']:,}",
            f"- Unique Complaint IDs: {integrity['unique_complaint_ids']:,}",
            "",
            "### Rows per Product",
            "",
            rows_per_product.to_frame().to_markdown(),
            "",
        ]

    lines += [
        "## Window coverage against Phase 2A counts",
        "",
        coverage.to_markdown(index=False),
        "",
    ]

    if failures:
        lines += [
            "## Failed windows",
            "",
            pd.DataFrame(
                [
                    {
                        "product": entry["product"],
                        "month": entry["month"],
                        "http_status": entry["http_status"],
                        "attempts": entry["attempts"],
                        "error": (entry["error"] or "")[:120],
                    }
                    for entry in failures
                ]
            ).to_markdown(index=False),
            "",
            "Failed windows are recorded rather than retried silently. Re-running "
            "the acquisition re-requests them.",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


def discover_raw_files(config: AcquisitionConfig, project_root: Path) -> list[Path]:
    """Find every raw CSV already present under the source directory."""
    source_dir = project_root / config.source_dir
    if not source_dir.exists():
        return []
    return sorted(source_dir.rglob("*.csv"))


def describe_raw_file(path: Path, project_root: Path) -> dict[str, Any]:
    """Measure a raw file that is already on disk, without re-requesting it."""
    frame = pd.read_csv(path, dtype=str)
    entry: dict[str, Any] = {
        "file_path": path.relative_to(project_root).as_posix(),
        "request_status": "loaded_from_disk",
        "file_size_bytes": path.stat().st_size,
        "sha256": file_checksum(path),
        "products": (
            sorted(frame["Product"].dropna().unique().tolist())
            if "Product" in frame.columns
            else []
        ),
    }
    entry.update(summarise_window(frame))
    return entry


def compare_coverage_to_plan(
    complaints: pd.DataFrame, plan: pd.DataFrame
) -> pd.DataFrame:
    """Check retrieved rows against the planned Phase 2A count, per product-month.

    Coverage is validated on the data itself rather than on request boundaries,
    so an extract assembled from wider windows - a manual export, for instance -
    is checked just as strictly as one assembled month by month.
    """
    observed = complaints.copy()
    observed["month"] = pd.to_datetime(
        observed["Date received"], errors="coerce"
    ).dt.strftime("%Y-%m")

    retrieved = (
        observed.groupby(["Product", "month"])
        .size()
        .rename("retrieved_rows")
        .reset_index()
        .rename(columns={"Product": "product"})
    )

    coverage = plan[["product", "month", "expected_rows"]].merge(
        retrieved, on=["product", "month"], how="outer", indicator=True
    )
    coverage["retrieved_rows"] = coverage["retrieved_rows"].fillna(0).astype(int)
    coverage["expected_rows"] = coverage["expected_rows"].fillna(0).astype(int)
    coverage["row_delta"] = coverage["retrieved_rows"] - coverage["expected_rows"]
    coverage["note"] = coverage["_merge"].map(
        {
            "both": "",
            "left_only": "planned but missing",
            "right_only": "outside the planned windows",
        }
    )
    return coverage.drop(columns="_merge").sort_values(["product", "month"])


def write_verification_report(
    path: Path,
    integrity: dict[str, Any],
    entries: list[dict[str, Any]],
    coverage: pd.DataFrame,
    checked_at: str,
) -> None:
    """Write the integrity report for an extract that was already on disk."""
    rows_per_product = pd.Series(
        integrity["rows_per_product"], name="rows"
    ).sort_values(ascending=False)
    rows_per_product.index.name = "product"

    missing = coverage[coverage["note"] == "planned but missing"]
    unplanned = coverage[coverage["note"] == "outside the planned windows"]

    lines = [
        "# Phase 2B - Raw Extract Verification",
        "",
        f"- Checked: {checked_at}",
        f"- Files inspected: {len(entries)}",
        "",
        "## Integrity checks",
        "",
        f"- Total rows: {integrity['total_rows']:,}",
        f"- Date received range: {integrity['min_date_received']} to "
        f"{integrity['max_date_received']}",
        f"- Missing narratives: {integrity['missing_narrative']:,}",
        f"- Duplicate Complaint IDs: {integrity['duplicate_complaint_ids']:,}",
        f"- Unique Complaint IDs: {integrity['unique_complaint_ids']:,}",
        "",
        "### Rows per Product",
        "",
        rows_per_product.to_frame().to_markdown(),
        "",
        "## Files",
        "",
        pd.DataFrame(
            [
                {
                    "file": entry["file_path"].rsplit("/", 1)[-1],
                    "rows": entry["retrieved_rows"],
                    "size_bytes": entry["file_size_bytes"],
                    "sha256": entry["sha256"][:16] + "...",
                }
                for entry in entries
            ]
        ).to_markdown(index=False),
        "",
        "## Coverage against the Phase 2A counts",
        "",
        f"- Product-months planned: {int((coverage['expected_rows'] > 0).sum())}",
        f"- Product-months missing: {len(missing)}",
        f"- Product-months outside the plan: {len(unplanned)}",
        "",
        coverage.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_verification(config_path: Path, project_root: Path) -> None:
    """Validate raw files already on disk and write the manifest and report.

    Used when the extract was obtained outside this script - for example a
    manual export - so that it passes exactly the same integrity checks and
    lands in the same audit trail.
    """
    config = load_config(config_path)
    counts = pd.read_csv(project_root / config.counts_csv)
    plan = plan_windows(counts, config)

    files = discover_raw_files(config, project_root)
    if not files:
        raise SystemExit(
            f"No CSV files found under {config.source_dir}. "
            "Place the raw exports there first."
        )

    logger.info("Inspecting %d raw file(s)", len(files))
    entries = [describe_raw_file(path, project_root) for path in files]
    complaints = pd.concat(
        [pd.read_csv(path, dtype=str) for path in files], ignore_index=True
    )

    integrity = run_integrity_checks(complaints)
    coverage = compare_coverage_to_plan(complaints, plan)
    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")

    manifest = build_manifest(config, plan, entries, integrity, checked_at)
    manifest["acquisition_method"] = (
        "Files already present on disk; verified rather than requested by this "
        "script. Record the actual retrieval route in the notes field."
    )
    manifest["notes"] = ""
    manifest["coverage"] = coverage.to_dict(orient="records")

    manifest_path = project_root / config.manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_path = project_root / config.integrity_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_verification_report(report_path, integrity, entries, coverage, checked_at)

    logger.info("Total rows: %s", f"{integrity['total_rows']:,}")
    logger.info("Saved manifest: %s", manifest_path)
    logger.info("Saved report: %s", report_path)


def run_acquisition(config_path: Path, project_root: Path, dry_run: bool) -> None:
    """Execute the Phase 2B raw acquisition end to end."""
    config = load_config(config_path)
    counts = pd.read_csv(project_root / config.counts_csv)
    plan = plan_windows(counts, config)

    logger.info(
        "Acquisition plan: %d windows, %s expected rows across %d products",
        len(plan),
        f"{plan['expected_rows'].sum():,}",
        plan["product"].nunique(),
    )

    if dry_run:
        summary = (
            plan.groupby("product")
            .agg(
                windows=("month", "count"),
                earliest_month=("month", "min"),
                latest_month=("month", "max"),
                expected_rows=("expected_rows", "sum"),
            )
            .sort_values("expected_rows", ascending=False)
        )
        print(summary.to_markdown())
        print(f"\nTotal expected rows: {plan['expected_rows'].sum():,}")
        return

    entries = acquire_windows(plan, config, project_root)
    retrieved_at = datetime.now().astimezone().isoformat(timespec="seconds")

    complaints = load_acquired_frame(entries, project_root)
    integrity = run_integrity_checks(complaints)

    manifest_path = project_root / config.manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config, plan, entries, integrity, retrieved_at)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_path = project_root / config.integrity_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_integrity_report(
        report_path, config, plan, entries, integrity, retrieved_at
    )

    logger.info("Total rows acquired: %s", f"{integrity.get('total_rows', 0):,}")
    logger.info("Saved manifest: %s", manifest_path)
    logger.info("Saved report: %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2B CFPB raw narrative acquisition."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the data acquisition config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned windows without contacting the API.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Skip acquisition and run the integrity checks, coverage comparison, "
            "and manifest over the raw files already under the source directory."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    project_root = Path(__file__).resolve().parents[2]
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if args.verify:
        run_verification(config_path, project_root)
    else:
        run_acquisition(config_path, project_root, args.dry_run)


if __name__ == "__main__":
    main()
