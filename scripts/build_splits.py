"""Builds the frozen train/val/test split. Run once.

Clustering the corpus for near-duplicates takes ~10 minutes, so the cluster ids
are written next to the split indices and every later run loads them instead of
recomputing. Re-running this script overwrites the frozen split, which
invalidates any result already produced against it.

    python -m scripts.build_splits [--recluster]
"""

import argparse
from pathlib import Path

import numpy as np

from src.data import LABEL_COL, deduplicate, load_raw
from src.dedup import CLUSTER_THRESHOLD, load_clusters, near_duplicate_clusters, save_clusters
from src.fingerprint import create_split_fingerprint
from src.logging_utils import get_logger
from src.split import create_stratified_split, save_splits

SPLIT_DIR = Path("data/splits")
CLUSTER_PATH = SPLIT_DIR / "near_dup_clusters.npy"
SEED = 42


def main(recluster: bool = False) -> None:
    logger = get_logger("build_splits")

    df = deduplicate(load_raw())
    logger.info(f"Deduplicated rows: {len(df):,}")

    if recluster or not CLUSTER_PATH.exists():
        logger.info(f"Clustering near-duplicates at cosine >= {CLUSTER_THRESHOLD} (~10 min)...")
        clusters = near_duplicate_clusters(df["Consumer complaint narrative"])
        save_clusters(clusters, CLUSTER_PATH)
    else:
        clusters = load_clusters(CLUSTER_PATH, expected_rows=len(df))

    sizes = np.bincount(clusters)
    logger.info(
        f"Clusters: {len(sizes):,} | singletons: {(sizes == 1).sum():,} | "
        f"rows in multi-row clusters: {sizes[sizes > 1].sum():,} | largest: {sizes.max():,}"
    )

    train_idx, val_idx, test_idx = create_stratified_split(df, clusters, seed=SEED)
    save_splits(train_idx, val_idx, test_idx, output_dir=SPLIT_DIR)
    create_split_fingerprint(df, train_idx, val_idx, test_idx).save(SPLIT_DIR / "split_manifest.json")

    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        logger.info(f"{name}: {len(idx):,} rows ({len(idx) / len(df):.4f})")
    logger.info(f"Class proportions held to within {_max_class_deviation(df, train_idx, val_idx, test_idx):.4f}")
    logger.info(f"Split frozen in {SPLIT_DIR}")


def _max_class_deviation(df, train_idx, val_idx, test_idx) -> float:
    overall = df[LABEL_COL].value_counts(normalize=True)
    deviations = [
        abs(df.iloc[idx][LABEL_COL].value_counts(normalize=True).get(label, 0.0) - share)
        for idx in (train_idx, val_idx, test_idx)
        for label, share in overall.items()
    ]
    return max(deviations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recluster",
        action="store_true",
        help="recompute near-duplicate clusters instead of loading the frozen ones",
    )
    main(**vars(parser.parse_args()))
