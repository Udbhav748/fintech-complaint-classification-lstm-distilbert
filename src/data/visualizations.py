"""Phase 4 publication-quality visualizations.

Generates:
  1. class_distribution.png: Overall 5-class dataset distribution.
  2. split_class_distribution.png: Train / Val / Test class distribution comparison.
  3. narrative_length_distribution.png: Word and character length histograms.
  4. distilbert_token_length_distribution.png: DistilBERT token length histogram and CDF.
  5. truncation_rate_comparison.png: Truncation rate comparison across candidate max lengths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Aesthetic styling constants
STYLE_PALETTE = ["#2b5c8f", "#3e8e7e", "#d97736", "#c0392b", "#7d5ba6"]
SPLIT_PALETTE = {"train": "#2b5c8f", "validation": "#3e8e7e", "test": "#d97736"}


def setup_plot_style() -> None:
    """Set global matplotlib styling for clear, readable figures."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#e5e5e5"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str = "product",
    output_path: Path | str = "reports/figures/phase4/class_distribution.png",
) -> Path:
    """Plot overall dataset class distribution with counts and percentages."""
    setup_plot_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    counts = df[target_col].value_counts()
    total = len(df)
    labels = counts.index.tolist()
    values = counts.values

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bars = ax.barh(labels, values, color=STYLE_PALETTE, height=0.65, edgecolor="none")
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        pct = (val / total) * 100
        ax.text(
            val + (total * 0.008),
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.2f}%)",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.set_title("CFPB Complaint Dataset: 5-Class Target Distribution (N = 107,992)", fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Number of Complaint Records", fontsize=11, labelpad=10)
    ax.set_xlim(0, max(values) * 1.22)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved class distribution plot: %s", out)
    return out


def plot_split_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "product",
    output_path: Path | str = "reports/figures/phase4/split_class_distribution.png",
) -> Path:
    """Plot Train / Validation / Test class distribution comparison."""
    setup_plot_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    splits = {
        "Train (70%)": train_df[target_col].value_counts(),
        "Validation (15%)": val_df[target_col].value_counts(),
        "Test (15%)": test_df[target_col].value_counts(),
    }

    df_splits = pd.DataFrame(splits).fillna(0)
    classes = df_splits.index.tolist()

    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    rects1 = ax.bar(x - width, df_splits["Train (70%)"], width, label=f"Train (N = {len(train_df):,})", color="#2b5c8f")
    rects2 = ax.bar(x, df_splits["Validation (15%)"], width, label=f"Validation (N = {len(val_df):,})", color="#3e8e7e")
    rects3 = ax.bar(x + width, df_splits["Test (15%)"], width, label=f"Test (N = {len(test_df):,})", color="#d97736")

    ax.set_title("Group-Aware Stratified Split: Class Distribution across Partitions", fontsize=13, pad=15, fontweight="bold")
    ax.set_ylabel("Complaint Records", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(", ", ",\n") for c in classes], fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved split class distribution plot: %s", out)
    return out


def plot_narrative_length_distribution(
    word_lengths: list[int],
    char_lengths: list[int],
    output_path: Path | str = "reports/figures/phase4/narrative_length_distribution.png",
) -> Path:
    """Plot word count and character count distributions with percentile markers."""
    setup_plot_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Word count plot
    w_arr = np.array(word_lengths)
    w_med = np.median(w_arr)
    w_p75 = np.percentile(w_arr, 75)
    w_p90 = np.percentile(w_arr, 90)

    ax1.hist(np.clip(w_arr, 0, 800), bins=50, color="#2b5c8f", alpha=0.85, edgecolor="white")
    ax1.axvline(w_med, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Median: {int(w_med)} words")
    ax1.axvline(w_p75, color="#d97736", linestyle=":", linewidth=1.5, label=f"75th %ile: {int(w_p75)} words")
    ax1.axvline(w_p90, color="#7d5ba6", linestyle="-.", linewidth=1.5, label=f"90th %ile: {int(w_p90)} words")
    ax1.set_title("Narrative Word Count Distribution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Word Count (clipped at 800)", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Character count plot
    c_arr = np.array(char_lengths)
    c_med = np.median(c_arr)
    c_p75 = np.percentile(c_arr, 75)

    ax2.hist(np.clip(c_arr, 0, 4500), bins=50, color="#3e8e7e", alpha=0.85, edgecolor="white")
    ax2.axvline(c_med, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Median: {int(c_med)} chars")
    ax2.axvline(c_p75, color="#d97736", linestyle=":", linewidth=1.5, label=f"75th %ile: {int(c_p75)} chars")
    ax2.set_title("Narrative Character Count Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Character Count (clipped at 4,500)", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved narrative length distribution plot: %s", out)
    return out


def plot_distilbert_token_length_distribution(
    token_lengths: list[int],
    candidate_lengths: list[int] = [64, 128, 256, 512],
    output_path: Path | str = "reports/figures/phase4/distilbert_token_length_distribution.png",
) -> Path:
    """Plot DistilBERT token length histogram and CDF with candidate max_length markers."""
    setup_plot_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t_arr = np.array(token_lengths)
    t_med = np.median(t_arr)
    t_p75 = np.percentile(t_arr, 75)
    t_p90 = np.percentile(t_arr, 90)

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)

    # Histogram
    n, bins, patches = ax1.hist(
        np.clip(t_arr, 0, 750),
        bins=60,
        color="#2b5c8f",
        alpha=0.75,
        edgecolor="white",
        label="Token Length Histogram",
    )
    ax1.set_xlabel("DistilBERT Sub-word Token Length (clipped at 750)", fontsize=11)
    ax1.set_ylabel("Number of Complaints", fontsize=11, color="#2b5c8f")
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")

    # Cumulative Distribution Function on secondary axis
    ax2 = ax1.twinx()
    sorted_data = np.sort(t_arr)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data)) * 100.0
    ax2.plot(sorted_data, yvals, color="#c0392b", linewidth=2.0, label="Cumulative Coverage (% retained)")
    ax2.set_ylabel("Cumulative % of Complaints Retained", fontsize=11, color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(0, 105)
    ax2.set_xlim(0, 750)

    # Candidate max_length vertical lines
    colors = ["#7d5ba6", "#3e8e7e", "#d97736", "#2c3e50"]
    for clen, col in zip(candidate_lengths, colors):
        retained = np.mean(t_arr <= clen) * 100.0
        ax1.axvline(clen, color=col, linestyle="--", linewidth=1.4, alpha=0.9)
        ax1.text(
            clen + 6,
            max(n) * 0.85,
            f"L={clen}\n({retained:.1f}% kept)",
            fontsize=8.5,
            fontweight="bold",
            color=col,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.9),
        )

    ax1.set_title("DistilBERT Token Length Distribution & Truncation Thresholds (distilbert-base-uncased)", fontsize=12, pad=15, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved DistilBERT token length distribution plot: %s", out)
    return out


def plot_truncation_rate_comparison(
    truncation_rates: dict[str, dict[str, Any]],
    output_path: Path | str = "reports/figures/phase4/truncation_rate_comparison.png",
) -> Path:
    """Plot bar chart comparing truncation and retention percentages for candidate max lengths."""
    setup_plot_style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lengths = [int(k) for k in truncation_rates.keys()]
    truncated_pcts = [truncation_rates[str(l)]["truncated_percentage"] for l in lengths]
    retained_pcts = [truncation_rates[str(l)]["retained_percentage"] for l in lengths]

    x = np.arange(len(lengths))
    width = 0.45

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    bars_ret = ax.bar(x, retained_pcts, width, label="Retained (% without truncation)", color="#3e8e7e")
    bars_trunc = ax.bar(x, truncated_pcts, width, bottom=retained_pcts, label="Truncated (%)", color="#c0392b", alpha=0.85)

    for i, (ret, trunc) in enumerate(zip(retained_pcts, truncated_pcts)):
        ax.text(i, ret / 2, f"{ret:.1f}% Kept", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        if trunc > 5.0:
            ax.text(i, ret + (trunc / 2), f"{trunc:.1f}% Trunc", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    ax.set_title("Sequence Retention vs Truncation by Candidate Max Length (DistilBERT)", fontsize=12, pad=15, fontweight="bold")
    ax.set_xlabel("Candidate max_length Setting", fontsize=11)
    ax.set_ylabel("Percentage of Total Complaints (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"max_length = {l}" for l in lengths], fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved truncation rate comparison plot: %s", out)
    return out
