"""Results infrastructure, schema validation, and comparison table generation.

Enforces:
- Canonical schema for results/runs.csv
- Strict validation against malformed, duplicated, or out-of-range results
- Stable experiment identifiers (M0, M1, M2, M3, M4, D0)
- Automated comparison table generation with exact delta calculations
"""

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from src.checkpoint import CheckpointMetadata, get_metadata_path, VALID_EXPERIMENTS
from src.evaluation import calculate_deltas, format_delta, format_metric, seed_statistics
from src.fingerprint import SplitFingerprint

RUNS_SCHEMA = [
    "experiment",
    "model",
    "seed",
    "macro_f1",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "best_epoch",
    "epochs_run",
    "train_time",
    "parameter_count",
    "checkpoint_path",
    "dataset_version",
]

DEFAULT_MODEL_DESCRIPTIONS = {
    "M0": "Unidirectional LSTM baseline, random embeddings",
    "M1": "Bidirectional LSTM",
    "M2": "BiLSTM + Spatial Dropout",
    "M3": "BiLSTM + Pretrained GloVe-100d",
    "M4": "BiLSTM + GloVe + LR schedule + EarlyStopping + max_len=256",
    "D0": "DistilBERT (fine-tuned transformer)",
}

REQUIRED_SEEDS_PER_EXPERIMENT = {
    "M0": 3,
    "M1": 1,
    "M2": 1,
    "M3": 1,
    "M4": 1,
    "D0": 3,
}


class ResultsValidationError(ValueError):
    """Raised when results/runs.csv is invalid, corrupt, or incomplete."""
    pass


def log_run_to_csv(
    run_dict: dict[str, Any],
    runs_csv_path: Union[str, Path] = "results/runs.csv",
) -> None:
    """Appends or updates a run in results/runs.csv while validating schema integrity."""
    p = Path(runs_csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Validate that run_dict has all required fields
    missing_fields = [k for k in RUNS_SCHEMA if k not in run_dict]
    if missing_fields:
        raise ValueError(f"Run dictionary is missing required fields: {missing_fields}")

    # Build single-row DataFrame
    filtered_dict = {k: run_dict[k] for k in RUNS_SCHEMA}
    new_df = pd.DataFrame([filtered_dict])

    if not p.exists():
        new_df.to_csv(p, index=False)
    else:
        existing_df = pd.read_csv(p)
        # Check if identical (experiment, seed) already exists -> update it
        mask = (existing_df["experiment"] == run_dict["experiment"]) & (
            existing_df["seed"] == run_dict["seed"]
        )
        if mask.any():
            existing_df = existing_df[~mask]
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_csv(p, index=False)

    # Run strict validation on the file
    validate_runs_csv(p, check_seed_completeness=False)


def validate_runs_csv(
    runs_csv_path: Union[str, Path] = "results/runs.csv",
    check_seed_completeness: bool = False,
    raise_on_error: bool = True,
) -> tuple[bool, list[str]]:
    """Strictly validates results/runs.csv against all project rules.

    Checks:
    1. File exists and is non-empty.
    2. Exact required columns are present.
    3. No null/blank values in critical fields.
    4. Experiment names belong strictly to VALID_EXPERIMENTS (M0–M4, D0; rejecting aliases).
    5. No duplicate (experiment, seed) entries.
    6. Metric ranges: macro_f1, accuracy, macro_precision, macro_recall in [0.0, 1.0].
    7. Epoch validity: 1 <= best_epoch <= epochs_run.
    8. Non-negative training times and parameter counts.
    9. Optional check: Required seed counts per experiment (3 for M0/D0, 1 for M1-M4).
    """
    p = Path(runs_csv_path)
    errors: list[str] = []

    if not p.exists():
        errors.append(f"Results file does not exist: {p}")
        if raise_on_error:
            raise ResultsValidationError(errors[0])
        return False, errors

    try:
        df = pd.read_csv(p)
    except Exception as e:
        errors.append(f"Cannot parse {p} as CSV: {e}")
        if raise_on_error:
            raise ResultsValidationError(errors[0])
        return False, errors

    if df.empty:
        errors.append(f"{p} is empty (0 rows).")
        if raise_on_error:
            raise ResultsValidationError(errors[0])
        return False, errors

    # Check schema
    missing_cols = [col for col in RUNS_SCHEMA if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns in {p}: {missing_cols}")

    # Check experiment naming and nulls
    if "experiment" in df.columns:
        null_exps = df["experiment"].isna().sum()
        if null_exps > 0:
            errors.append(f"Found {null_exps} rows with null/missing 'experiment'")
        
        invalid_names = set(df["experiment"].dropna()) - set(VALID_EXPERIMENTS)
        if invalid_names:
            errors.append(
                f"Invalid experiment names detected: {invalid_names}. "
                f"Must strictly be one of {VALID_EXPERIMENTS} (no aliases allowed)."
            )

    # Check duplicate (experiment, seed)
    if "experiment" in df.columns and "seed" in df.columns:
        dups = df.duplicated(subset=["experiment", "seed"], keep=False)
        if dups.any():
            dup_pairs = df[dups][["experiment", "seed"]].to_dict(orient="records")
            errors.append(f"Duplicate (experiment, seed) entries found: {dup_pairs}")

    # Check metric ranges
    metric_cols = ["macro_f1", "accuracy", "macro_precision", "macro_recall"]
    for col in metric_cols:
        if col in df.columns:
            invalid_range = (df[col] < 0.0) | (df[col] > 1.0)
            if invalid_range.any():
                bad_vals = df[invalid_range][["experiment", "seed", col]].to_dict(orient="records")
                errors.append(f"Metric '{col}' out of valid [0, 1] range: {bad_vals}")

    # Check epochs
    if "best_epoch" in df.columns and "epochs_run" in df.columns:
        bad_best = df["best_epoch"] < 1
        if bad_best.any():
            errors.append(f"Rows with best_epoch < 1 found: {df[bad_best][['experiment', 'seed', 'best_epoch']].to_dict(orient='records')}")

        bad_epochs = df["best_epoch"] > df["epochs_run"]
        if bad_epochs.any():
            errors.append(
                f"Rows where best_epoch > epochs_run found: {df[bad_epochs][['experiment', 'seed', 'best_epoch', 'epochs_run']].to_dict(orient='records')}"
            )

    # Check train_time and parameter_count
    if "train_time" in df.columns:
        bad_time = df["train_time"] < 0
        if bad_time.any():
            errors.append(f"Rows with negative train_time found")

    if "parameter_count" in df.columns:
        bad_params = df["parameter_count"] <= 0
        if bad_params.any():
            errors.append(f"Rows with parameter_count <= 0 found")

    # Check seed completeness if requested
    if check_seed_completeness and "experiment" in df.columns:
        counts = df["experiment"].value_counts().to_dict()
        for exp, req_seeds in REQUIRED_SEEDS_PER_EXPERIMENT.items():
            actual_seeds = counts.get(exp, 0)
            if actual_seeds < req_seeds:
                errors.append(
                    f"Experiment {exp} requires {req_seeds} seeds, but only {actual_seeds} found in results."
                )

    is_valid = len(errors) == 0
    if not is_valid and raise_on_error:
        raise ResultsValidationError(
            "results/runs.csv validation failed:\n" + "\n".join(f"- {e}" for e in errors)
        )

    return is_valid, errors


def generate_comparison_table(
    runs_csv_path: Union[str, Path] = "results/runs.csv",
    model_descriptions: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Generates the official project comparison table directly from results/runs.csv.

    Columns:
    - Model (M0–M4, D0)
    - Configuration (descriptive text)
    - Macro-F1 (formatted mean ± std for multi-seed, or single float)
    - Accuracy (formatted mean ± std for multi-seed, or single float)
    - Δ vs Previous (calculated unrounded, formatted with explicit sign)
    - Δ vs M0 (calculated unrounded, formatted with explicit sign)
    - Interpretation (placeholder for documented findings)
    """
    p = Path(runs_csv_path)
    if not p.exists():
        return pd.DataFrame(
            columns=["Model", "Configuration", "Macro-F1", "Accuracy", "Δ vs Previous", "Δ vs M0", "Interpretation"]
        )

    validate_runs_csv(p, check_seed_completeness=False, raise_on_error=True)
    df = pd.read_csv(p)

    descriptions = model_descriptions or DEFAULT_MODEL_DESCRIPTIONS
    rows = []
    f1_means: dict[str, float] = {}

    for i, exp in enumerate(VALID_EXPERIMENTS):
        sub = df[df["experiment"] == exp]
        if sub.empty:
            continue

        desc = descriptions.get(exp, "")

        f1_stats = seed_statistics(sub["macro_f1"].values)
        acc_stats = seed_statistics(sub["accuracy"].values)
        mean_f1 = f1_stats["mean"]
        f1_means[exp] = mean_f1

        f1_str = format_metric(mean_f1, f1_stats["std"])
        acc_str = format_metric(acc_stats["mean"], acc_stats["std"])

        # Calculate Deltas
        prev_exp = VALID_EXPERIMENTS[i - 1] if i > 0 else None
        prev_f1 = f1_means.get(prev_exp) if prev_exp else None
        m0_f1 = f1_means.get("M0")

        deltas = calculate_deltas(
            current_f1=mean_f1,
            previous_f1=prev_f1,
            baseline_m0_f1=m0_f1 if exp != "M0" else None,
        )

        delta_prev_str = format_delta(deltas["delta_vs_previous"])
        delta_m0_str = format_delta(deltas["delta_vs_m0"])

        rows.append({
            "Model": exp,
            "Configuration": desc,
            "Macro-F1": f1_str,
            "Accuracy": acc_str,
            "Δ vs Previous": delta_prev_str,
            "Δ vs M0": delta_m0_str,
            "Interpretation": "",
        })

    return pd.DataFrame(rows)


def verify_run_traceability(
    run_dict: dict[str, Any],
    split_manifest_path: Union[str, Path] = "data/splits/split_manifest.json",
    checkpoint_dir: Union[str, Path] = "checkpoints",
) -> tuple[bool, list[str]]:
    """Checks that a result row can actually be traced back to what produced it.

    `RUNS_SCHEMA`'s `dataset_version` and `checkpoint_path` are the two fields
    that carry provenance; this function is what makes them mean something,
    rather than being strings nobody checks. Verifies:

    1. `dataset_version` equals the frozen split's `dataset_content_sha256` -
       the row describes a result on *this* dataset/split, not some other one.
    2. `checkpoint_path` has a matching `CheckpointMetadata` record (written by
       `src.checkpoint.save_checkpoint_record`) whose `experiment`/`seed` agree
       with the row's own `experiment`/`seed`, and whose `monitor_metric` is
       `val_macro_f1` - the model-selection policy this project requires.

    Does not check `git_commit` or `preprocessing_version` as run-record columns:
    those are captured separately (`src.reproducibility.EnvironmentInfo`,
    `artifacts/preprocessing/preprocessing_version.json`) rather than duplicated
    into every row of `results/runs.csv`, consistent with "use the existing
    project schema" - `RUNS_SCHEMA` is not extended here.
    """
    errors: list[str] = []

    expected_hash = SplitFingerprint.load(split_manifest_path).dataset_content_sha256
    if run_dict.get("dataset_version") != expected_hash:
        errors.append(
            f"dataset_version '{run_dict.get('dataset_version')}' does not match the frozen "
            f"split's dataset_content_sha256 '{expected_hash}'"
        )

    experiment = run_dict.get("experiment")
    seed = run_dict.get("seed")
    meta_path = get_metadata_path(experiment, seed, checkpoint_dir=checkpoint_dir)
    if not meta_path.exists():
        errors.append(f"No checkpoint metadata found for {experiment} seed={seed} at {meta_path}")
    else:
        meta = CheckpointMetadata.load(meta_path)
        if meta.experiment != experiment or meta.seed != seed:
            errors.append(
                f"Checkpoint metadata at {meta_path} is for {meta.experiment} seed={meta.seed}, "
                f"not {experiment} seed={seed}"
            )
        if meta.monitor_metric != "val_macro_f1":
            errors.append(f"Checkpoint metadata monitor_metric is '{meta.monitor_metric}', not 'val_macro_f1'")

    return len(errors) == 0, errors
