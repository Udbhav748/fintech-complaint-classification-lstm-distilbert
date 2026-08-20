"""Comprehensive end-to-end system verification script.

Verifies:
1. Dataset loading, deduplication, and fingerprinting
2. Split generation, persistence, and strict invariant validation
3. Evaluation layer precision against scikit-learn
4. Checkpoint management and metric enforcement
5. Results logging, validation, and automated table generation
6. Environment metadata capture
"""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.checkpoint import (
    CheckpointValidationError,
    get_checkpoint_path,
    save_checkpoint_record,
    verify_checkpoint,
)
from src.config import get_experiment_config, load_data_config
from src.data import LABELS, deduplicate, load_raw
from src.evaluation import calculate_deltas, compute_metrics, format_delta
from src.fingerprint import (
    create_dataset_fingerprint,
    create_split_fingerprint,
    verify_dataset_fingerprint,
    verify_split_fingerprint,
)
from src.logging_utils import get_logger
from src.reproducibility import EnvironmentInfo, set_seed
from src.results import (
    generate_comparison_table,
    log_run_to_csv,
    validate_runs_csv,
)
from src.dedup import load_clusters
from src.split import (
    create_stratified_split,
    load_splits,
    save_splits,
    validate_split,
)


def run_full_verification() -> bool:
    logger = get_logger("verification")
    logger.info("=== STARTING FULL SYSTEM VERIFICATION ===")

    # 1. Reproducibility & Environment Info
    logger.info("[1/6] Verifying Reproducibility & Environment Info...")
    set_seed(42)
    env = EnvironmentInfo.capture()
    assert env.python_version, "Python version missing"
    assert "pandas" in env.packages, "Pandas package missing"
    logger.info(f"  Python: {env.python_version} | Git Commit: {env.git_commit[:8] if env.git_commit != 'unknown' else 'N/A'}")
    logger.info(f"  Device: {env.device.get('device_type')} ({env.device.get('device_name')})")

    # 2. Data Loading & Deduplication
    logger.info("[2/6] Verifying Dataset & Fingerprint...")
    df_raw = load_raw(Path("data/combined_complaints.parquet"))
    assert len(df_raw) == 107992, f"Expected 107,992 rows, got {len(df_raw)}"
    assert len(df_raw.columns) == 16, f"Expected 16 columns, got {len(df_raw.columns)}"

    df_dedup = deduplicate(df_raw)
    assert len(df_dedup) == 101802, f"Expected 101,802 rows, got {len(df_dedup)}"
    logger.info(f"  Raw rows: {len(df_raw):,} | Deduplicated rows: {len(df_dedup):,}")

    manifest_path = Path("data/dataset_manifest.json")
    is_fp_valid, fp_mismatches = verify_dataset_fingerprint(df_raw, manifest_path)
    assert is_fp_valid, f"Dataset manifest mismatch: {fp_mismatches}"
    logger.info(f"  Dataset Manifest verification PASSED.")

    # 3. Split Creation & Validation
    logger.info("[3/6] Verifying Split Generation & Invariants...")
    split_dir = Path("data/splits")
    clusters = load_clusters(split_dir / "near_dup_clusters.npy", expected_rows=len(df_dedup))
    train_idx, val_idx, test_idx = create_stratified_split(df_dedup, clusters, seed=42)
    save_splits(train_idx, val_idx, test_idx, output_dir=split_dir)
    loaded_tr, loaded_v, loaded_te = load_splits(split_dir=split_dir)

    # Exact counts shift by a few rows with the near-duplicate grouping, so the
    # proportions are asserted instead of frozen row counts.
    total = len(df_dedup)
    for name, idx, target in (("train", loaded_tr, 0.80), ("val", loaded_v, 0.10), ("test", loaded_te, 0.10)):
        share = len(idx) / total
        assert abs(share - target) < 0.005, f"{name} split is {share:.4f} of the data, expected ~{target}"

    validate_split(df_dedup, loaded_tr, loaded_v, loaded_te, clusters)
    split_fp = create_split_fingerprint(df_dedup, loaded_tr, loaded_v, loaded_te)
    split_fp.save(split_dir / "split_manifest.json")
    is_split_valid, split_mismatches = verify_split_fingerprint(df_dedup, loaded_tr, loaded_v, loaded_te, split_dir / "split_manifest.json")
    assert is_split_valid, f"Split manifest verification failed: {split_mismatches}"
    logger.info(f"  Split files persisted and verified: train={len(loaded_tr):,}, val={len(loaded_v):,}, test={len(loaded_te):,}")

    # 4. Evaluation Layer vs Scikit-Learn
    logger.info("[4/6] Verifying Evaluation Layer vs Scikit-Learn...")
    np.random.seed(42)
    sample_true = np.random.randint(0, 5, size=1000)
    sample_pred = sample_true.copy()
    mask = np.random.rand(1000) < 0.2
    sample_pred[mask] = np.random.randint(0, 5, size=int(mask.sum()))

    eval_out = compute_metrics(sample_true, sample_pred, labels=LABELS)
    sklearn_f1 = f1_score(sample_true, sample_pred, average="macro", zero_division=0)
    assert np.isclose(eval_out["macro_f1"], sklearn_f1), f"Macro-F1 mismatch: custom={eval_out['macro_f1']}, sklearn={sklearn_f1}"

    deltas = calculate_deltas(current_f1=0.8580, previous_f1=0.8420, baseline_m0_f1=0.8420)
    assert format_delta(deltas["delta_vs_previous"]) == "+0.0160"
    assert format_delta(deltas["delta_vs_m0"]) == "+0.0160"
    logger.info(f"  Macro-F1 custom == sklearn: verified ({eval_out['macro_f1']:.6f} == {sklearn_f1:.6f})")

    # 5. Checkpoint Management
    logger.info("[5/6] Verifying Checkpoint Management...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt_file = get_checkpoint_path("M0", seed=42, checkpoint_dir=tmp_path)
        ckpt_file.write_bytes(b"dummy_best_weights_12345")
        save_checkpoint_record(
            experiment="M0",
            seed=42,
            best_epoch=7,
            best_val_macro_f1=0.8420,
            checkpoint_file=ckpt_file,
            checkpoint_dir=tmp_path,
        )
        is_ckpt_valid, ckpt_errors = verify_checkpoint("M0", seed=42, checkpoint_dir=tmp_path)
        assert is_ckpt_valid, f"Checkpoint verification failed: {ckpt_errors}"
    logger.info("  Checkpoint pathing and metadata verified.")

    # 6. Results Infrastructure & Comparison Table
    logger.info("[6/6] Verifying Results Logging & Comparison Table...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_csv = Path(tmp) / "runs.csv"
        # Log 3 seeds for M0
        for seed, f1 in [(42, 0.8410), (123, 0.8430), (456, 0.8420)]:
            log_run_to_csv({
                "experiment": "M0",
                "model": "LSTM baseline",
                "seed": seed,
                "macro_f1": f1,
                "accuracy": f1 + 0.003,
                "macro_precision": f1 + 0.001,
                "macro_recall": f1 - 0.001,
                "best_epoch": 7,
                "epochs_run": 10,
                "train_time": 120.0,
                "parameter_count": 150000,
                "checkpoint_path": f"checkpoints/M0_seed{seed}.weights.h5",
                "dataset_version": "149a4b98ba4d4bbcb1f15ca74da93bf4529be9755a22b19603d85f93e69df6a4",
            }, runs_csv_path=tmp_csv)

        # Log 1 seed for M1
        log_run_to_csv({
            "experiment": "M1",
            "model": "Bidirectional LSTM",
            "seed": 42,
            "macro_f1": 0.8580,
            "accuracy": 0.8610,
            "macro_precision": 0.8590,
            "macro_recall": 0.8570,
            "best_epoch": 6,
            "epochs_run": 10,
            "train_time": 150.0,
            "parameter_count": 300000,
            "checkpoint_path": "checkpoints/M1_seed42.weights.h5",
            "dataset_version": "149a4b98ba4d4bbcb1f15ca74da93bf4529be9755a22b19603d85f93e69df6a4",
        }, runs_csv_path=tmp_csv)

        table = generate_comparison_table(runs_csv_path=tmp_csv)
        assert len(table) == 2, f"Expected 2 rows in table, got {len(table)}"
        assert table.iloc[1]["Δ vs Previous"] == "+0.0160"
        assert table.iloc[1]["Δ vs M0"] == "+0.0160"

    logger.info("  Results logging & comparison table verified.")
    logger.info("\n>>> ALL SYSTEM VERIFICATIONS PASSED SUCCESSFULLY <<<")
    return True


if __name__ == "__main__":
    success = run_full_verification()
    sys.exit(0 if success else 1)
