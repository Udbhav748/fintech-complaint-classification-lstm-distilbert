"""Orchestrates the M0 baseline: 3 seeds, sequential, on Kaggle GPU, each one
validated before the next starts. Resumable - a seed with a valid completed run
already registered is skipped, never retrained silently.

Requires the Kaggle dataset built by scripts/kaggle_package_m0.py to already
exist (run that first, once, or after any frozen input changes - it never should
change mid-M0).

    python scripts/run_m0.py
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.checkpoint import get_checkpoint_path, get_metadata_path, verify_checkpoint
from src.config import get_experiment_config
from src.evaluation import seed_statistics
from src.fingerprint import SplitFingerprint
from src.reproducibility import get_git_info
from src.results import RUNS_SCHEMA, log_run_to_csv, validate_runs_csv

EXPERIMENT = "M0"
DATASET_SLUG = "cfpb-m0-frozen-inputs"
KAGGLE_USER = "udbhav748"
TRAIN_TEMPLATE = REPO_ROOT / "scripts" / "kaggle_train_m0.py"
STAGING_DIR = REPO_ROOT / "kaggle_staging"
RUNS_DIR = REPO_ROOT / "kaggle_runs"
RESULTS_DIR = REPO_ROOT / "results" / "m0"
RUNS_CSV = REPO_ROOT / "results" / "runs.csv"
SPLIT_MANIFEST = REPO_ROOT / "data" / "splits" / "split_manifest.json"

# project_plan.md §9: "If sigma > 0.8 F1 points, most ladder deltas are
# unresolvable" - reused as-is, not a new threshold invented for this task.
STABILITY_THRESHOLD_F1_POINTS = 0.008

POLL_INTERVAL_SECONDS = 30
POLL_TIMEOUT_SECONDS = 3600


class RunFailure(RuntimeError):
    pass


def seed_already_done(seed: int) -> bool:
    """Resume safety: a seed only counts as done if it is registered AND its
    checkpoint still exists and loads - not just present in runs.csv."""
    if not RUNS_CSV.exists():
        return False
    df = pd.read_csv(RUNS_CSV)
    match = df[(df["experiment"] == EXPERIMENT) & (df["seed"] == seed)]
    if match.empty:
        return False
    is_valid, _ = verify_checkpoint(EXPERIMENT, seed=seed, checkpoint_dir=REPO_ROOT / "checkpoints", raise_on_failure=False)
    return is_valid


def kernel_slug(seed: int) -> str:
    # Kaggle derives the actual kernel slug from the title when creating a new
    # kernel, ignoring a mismatched explicit `id` (learned the hard way: a title
    # of "CFPB M0 seed 42" silently became slug "cfpb-m0-seed-42", not the
    # "cfpb-m0-seed42" the `id` field asked for). The title is set to this exact
    # string below, so the two can never diverge again.
    return f"cfpb-m0-seed-{seed}"


def push_kernel(seed: int) -> Path:
    stage = STAGING_DIR / kernel_slug(seed)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    script = TRAIN_TEMPLATE.read_text(encoding="utf-8").replace("{{SEED}}", str(seed))
    (stage / "kaggle_train_m0.py").write_text(script, encoding="utf-8")

    metadata = {
        "id": f"{KAGGLE_USER}/{kernel_slug(seed)}",
        "title": kernel_slug(seed),
        "code_file": "kaggle_train_m0.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [f"{KAGGLE_USER}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    result = subprocess.run(["kaggle", "kernels", "push", "-p", str(stage)], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RunFailure(f"kaggle kernels push failed for seed {seed}:\n{result.stderr}")
    return stage


def poll_until_done(seed: int) -> str:
    ref = f"{KAGGLE_USER}/{kernel_slug(seed)}"
    waited = 0
    while waited < POLL_TIMEOUT_SECONDS:
        result = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True, text=True)
        if result.returncode != 0:
            raise RunFailure(f"kaggle kernels status failed for seed {seed}:\n{result.stderr}")
        status_line = result.stdout.strip()
        print(f"[{waited}s] {status_line}")
        if "COMPLETE" in status_line:
            return status_line
        if "ERROR" in status_line or "CANCEL" in status_line:
            raise RunFailure(f"Kaggle kernel run for seed {seed} did not complete: {status_line}")
        time.sleep(POLL_INTERVAL_SECONDS)
        waited += POLL_INTERVAL_SECONDS
    raise RunFailure(f"Timed out after {POLL_TIMEOUT_SECONDS}s waiting for seed {seed}")


def download_output(seed: int) -> Path:
    ref = f"{KAGGLE_USER}/{kernel_slug(seed)}"
    out_dir = RUNS_DIR / kernel_slug(seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    result = subprocess.run(
        ["kaggle", "kernels", "output", ref, "-p", str(out_dir)], capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RunFailure(f"kaggle kernels output failed for seed {seed}:\n{result.stderr}")
    return out_dir


def validate_run(seed: int, out_dir: Path) -> list[str]:
    """Part 7's checks, run on structured downloaded files only - never on
    console log text. Returns a list of problems; empty means the run is trusted."""
    errors = []

    record_path = out_dir / "run_record.json"
    if not record_path.exists():
        return [f"run_record.json missing from {out_dir}"]
    record = json.loads(record_path.read_text(encoding="utf-8"))

    missing = [k for k in RUNS_SCHEMA if k not in record]
    if missing:
        errors.append(f"run_record.json missing fields: {missing}")

    if record.get("experiment") != EXPERIMENT:
        errors.append(f"experiment is '{record.get('experiment')}', expected '{EXPERIMENT}'")
    if record.get("seed") != seed:
        errors.append(f"seed is {record.get('seed')}, expected {seed}")

    for metric in ("macro_f1", "accuracy", "macro_precision", "macro_recall"):
        val = record.get(metric)
        if val is None or not (0.0 <= val <= 1.0):
            errors.append(f"{metric}={val} is out of [0, 1]")

    best_epoch, epochs_run = record.get("best_epoch"), record.get("epochs_run")
    max_epochs = get_experiment_config(EXPERIMENT)["epochs"]
    if not (best_epoch and 1 <= best_epoch <= epochs_run):
        errors.append(f"best_epoch={best_epoch} not in [1, epochs_run={epochs_run}]")
    if not epochs_run or epochs_run > max_epochs:
        errors.append(f"epochs_run={epochs_run} exceeds the locked budget of {max_epochs}")

    expected_hash = SplitFingerprint.load(SPLIT_MANIFEST).dataset_content_sha256
    if record.get("dataset_version") != expected_hash:
        errors.append("dataset_version does not match the local frozen split manifest")

    val_check_path = out_dir / "val_check.json"
    if not val_check_path.exists():
        errors.append("val_check.json missing - can't confirm checkpoint selection was verified")
    else:
        agreement = json.loads(val_check_path.read_text(encoding="utf-8"))["agreement"]
        if agreement > 1e-3:
            errors.append(f"val_macro_f1 agreement check failed on Kaggle: {agreement}")

    ckpt_dir = out_dir / "checkpoints"
    ckpt_files = list(ckpt_dir.glob(f"{EXPERIMENT}_seed{seed}.weights.h5")) if ckpt_dir.exists() else []
    if not ckpt_files or ckpt_files[0].stat().st_size == 0:
        errors.append("checkpoint .weights.h5 missing or empty in downloaded output")
    else:
        try:
            from src.models import RecurrentModelConfig, build_recurrent_model
            model = build_recurrent_model(RecurrentModelConfig.from_experiment(EXPERIMENT))
            model.load_weights(str(ckpt_files[0]))
        except Exception as e:
            errors.append(f"checkpoint failed to load into a freshly-built M0 model: {e}")

    for fname in ("history.json", "test_predictions.csv", "test_metrics.json", "environment.json"):
        if not (out_dir / fname).exists():
            errors.append(f"expected output file missing: {fname}")

    # Structural leakage guard: the training script's test-evaluation code runs
    # only after model.load_weights(checkpoint_path) and the val-only agreement
    # check, and ModelCheckpoint monitors val_macro_f1, never a test metric - so
    # there is no code path where test data could influence which epoch was
    # selected. Nothing further to check at the artifact level.

    return errors


def register_run(seed: int, out_dir: Path) -> None:
    record = json.loads((out_dir / "run_record.json").read_text(encoding="utf-8"))

    checkpoints_dir = REPO_ROOT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    src_ckpt = out_dir / "checkpoints" / f"{EXPERIMENT}_seed{seed}.weights.h5"
    src_meta = out_dir / "checkpoints" / f"{EXPERIMENT}_seed{seed}_meta.json"
    local_ckpt_path = get_checkpoint_path(EXPERIMENT, seed, checkpoints_dir)
    shutil.copy2(src_ckpt, local_ckpt_path)
    shutil.copy2(src_meta, get_metadata_path(EXPERIMENT, seed, checkpoints_dir))

    # The kernel recorded its own /kaggle/working path, which stops existing
    # once the session ends - repoint the persisted record at where the
    # checkpoint actually lives now, or "traceable" would be a lie in a week.
    record["checkpoint_path"] = str(local_ckpt_path)
    log_run_to_csv(record, runs_csv_path=RUNS_CSV)

    seed_dir = RESULTS_DIR / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("history.json", "test_predictions.csv", "test_metrics.json", "environment.json", "val_check.json"):
        shutil.copy2(out_dir / fname, seed_dir / fname)

    preprocessing_version = json.loads(
        (REPO_ROOT / "artifacts" / "preprocessing" / "preprocessing_version.json").read_text(encoding="utf-8")
    )["version_id"]
    manifest = {
        "experiment": EXPERIMENT, "seed": seed,
        "git_commit": get_git_info(REPO_ROOT)["git_commit"],
        "kaggle_dataset": f"{KAGGLE_USER}/{DATASET_SLUG}",
        "kaggle_kernel": f"{KAGGLE_USER}/{kernel_slug(seed)}",
        "dataset_content_sha256": record["dataset_version"],
        "preprocessing_version": preprocessing_version,
    }
    (seed_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"seed {seed} registered: macro_f1={record['macro_f1']:.4f} best_epoch={record['best_epoch']}")


def run_seed(seed: int) -> None:
    print(f"\n=== M0 seed {seed} ===")
    push_kernel(seed)
    poll_until_done(seed)
    out_dir = download_output(seed)
    errors = validate_run(seed, out_dir)
    if errors:
        report_path = RESULTS_DIR / f"seed{seed}_FAILED.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"seed": seed, "errors": errors}, indent=2), encoding="utf-8")
        raise RunFailure(f"seed {seed} failed validation ({len(errors)} issue(s)) - see {report_path}:\n" + "\n".join(errors))
    register_run(seed, out_dir)


def stability_report(seeds: list[int]) -> None:
    df = pd.read_csv(RUNS_CSV)
    rows = df[(df["experiment"] == EXPERIMENT) & (df["seed"].isin(seeds))]
    f1_values = rows["macro_f1"].tolist()
    stats = seed_statistics(f1_values)

    print("\n=== M0 stability (3 seeds, stability reference - not a significance test) ===")
    for _, r in rows.iterrows():
        print(f"  seed {int(r['seed'])}: macro_f1={r['macro_f1']:.4f}")
    print(f"  mean = {stats['mean']:.4f}, std (ddof=1) = {stats['std']:.4f}, n={stats['n']}")
    print(f"  min = {min(f1_values):.4f}, max = {max(f1_values):.4f}, spread = {max(f1_values) - min(f1_values):.4f}")

    if stats["std"] is not None and stats["std"] > STABILITY_THRESHOLD_F1_POINTS:
        print(
            f"  ** std {stats['std']:.4f} exceeds the pre-registered {STABILITY_THRESHOLD_F1_POINTS} "
            f"threshold (project_plan.md §9) - most ladder deltas would be unresolvable. STOP and investigate "
            f"before proceeding to M1. **"
        )
    else:
        print(f"  std within the pre-registered {STABILITY_THRESHOLD_F1_POINTS} threshold.")


def main() -> int:
    seeds = get_experiment_config(EXPERIMENT)["seeds"]
    print(f"M0 seeds (from src.config, not invented): {seeds}")

    completed = []
    for seed in seeds:
        if seed_already_done(seed):
            print(f"seed {seed}: already validated and registered - skipping (resume safety)")
            completed.append(seed)
            continue
        try:
            run_seed(seed)
            completed.append(seed)
        except RunFailure as e:
            print(f"\nSTOPPING: {e}")
            print(f"Seeds completed before this failure: {completed}")
            return 1

    is_valid, errors = validate_runs_csv(RUNS_CSV, check_seed_completeness=False)
    if not is_valid:
        print(f"results/runs.csv failed validation after registration: {errors}")
        return 1

    stability_report(completed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
