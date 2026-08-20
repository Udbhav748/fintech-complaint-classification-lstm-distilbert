#!/usr/bin/env python
"""Pre-flight engineering guard and project validation script.

Run this script before executing experiments to ensure that core project assets,
experiment configurations, split manifests, and unit tests remain intact.
"""

from pathlib import Path
import subprocess
import sys
import unittest

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.checkpoint import VALID_EXPERIMENTS
from src.config import EXPERIMENT_CONFIGS, get_experiment_config, load_data_config
from src.fingerprint import DatasetFingerprint, SplitFingerprint
from src.results import RUNS_SCHEMA, validate_runs_csv


def check_project_structure() -> bool:
    """Verifies that all core directories and required files exist."""
    required_paths = [
        Path("configs/data_config.json"),
        Path("data/combined_complaints.parquet"),
        Path("data/dataset_manifest.json"),
        Path("data/splits/split_manifest.json"),
        Path("data/splits/train_idx.npy"),
        Path("data/splits/val_idx.npy"),
        Path("data/splits/test_idx.npy"),
        Path("project_plan.md"),
        Path("README.md"),
        Path("requirements.txt"),
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print(f"[FAIL] project structure — Missing files: {missing}")
        return False
    print("[OK] project structure")
    return True


def check_required_notebooks() -> bool:
    """Verifies that the modeling and EDA notebooks exist."""
    notebooks = [
        Path("notebooks/01_eda.ipynb"),
    ]
    missing = [str(p) for p in notebooks if not p.exists()]
    if missing:
        print(f"[FAIL] required notebooks — Missing: {missing}")
        return False
    print("[OK] required notebooks")
    return True


def check_config_integrity() -> bool:
    """Verifies central data and model configurations."""
    try:
        cfg = load_data_config()
        assert cfg["dataset"]["rows_raw"] == 107992
        assert cfg["dataset"]["rows_after_dedup"] == 101802

        # Check that all M0-M4, D0 configs exist
        for exp in VALID_EXPERIMENTS:
            m_cfg = get_experiment_config(exp)
            assert m_cfg["experiment"] == exp
            assert m_cfg["learning_rate"] > 0
            assert m_cfg["epochs"] >= 1
    except Exception as e:
        print(f"[FAIL] config — Error: {e}")
        return False

    print("[OK] config")
    return True


def check_split_manifest() -> bool:
    """Verifies split manifest indices and size consistency."""
    try:
        split_fp = SplitFingerprint.load("data/splits/split_manifest.json")
        assert split_fp.total_count == 101802, f"Expected 101,802, got {split_fp.total_count}"
        assert split_fp.train_count + split_fp.val_count + split_fp.test_count == split_fp.total_count
        assert split_fp.disjoint_indices_verified is True
        assert split_fp.zero_duplicate_cross_split_verified is True
    except Exception as e:
        print(f"[FAIL] split manifest — Error: {e}")
        return False

    print("[OK] split manifest")
    return True


def check_experiment_ids() -> bool:
    """Verifies that project_plan.md and configs use the exact canonical identifiers."""
    expected_ids = {"M0", "M1", "M2", "M3", "M4", "D0"}
    if set(VALID_EXPERIMENTS) != expected_ids:
        print(f"[FAIL] experiment IDs — Mismatch: {set(VALID_EXPERIMENTS)} != {expected_ids}")
        return False

    # Check project_plan.md mentions all IDs
    plan_text = Path("project_plan.md").read_text(encoding="utf-8")
    missing_in_plan = [exp for exp in expected_ids if exp not in plan_text]
    if missing_in_plan:
        print(f"[FAIL] experiment IDs — Missing in project_plan.md: {missing_in_plan}")
        return False

    print("[OK] experiment IDs")
    return True


def check_results_schema() -> bool:
    """Verifies results/runs.csv schema integrity if the file exists."""
    runs_csv = Path("results/runs.csv")
    if runs_csv.exists() and runs_csv.stat().st_size > 0:
        is_valid, errors = validate_runs_csv(runs_csv, raise_on_error=False)
        if not is_valid:
            print(f"[FAIL] results schema — Validation errors: {errors}")
            return False
    print("[OK] results schema")
    return True


def check_tests() -> bool:
    """Executes the test suite silently and reports status."""
    loader = unittest.TestLoader()
    suite = loader.discover(str(REPO_ROOT / "tests"))
    runner = unittest.TextTestRunner(stream=open(Path(tempfile_dir := tempfile_or_devnull()), "w"), verbosity=0)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print(f"[FAIL] tests — {len(result.failures)} failures, {len(result.errors)} errors")
        return False
    print(f"[OK] tests ({result.testsRun} passed)")
    return True


def tempfile_or_devnull() -> str:
    import tempfile
    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()
    return f.name


def check_git_state() -> None:
    """Reports git branch and working tree state."""
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        dirty = " [uncommitted changes present]" if status else " [clean]"
        print(f"[INFO] git branch: {branch}{dirty}")
    except Exception:
        pass


def main() -> int:
    print("=" * 50)
    print("CFPB Classification — Pre-flight Project Guard")
    print("=" * 50)
    check_git_state()
    print("-" * 50)

    checks = [
        check_project_structure,
        check_required_notebooks,
        check_config_integrity,
        check_split_manifest,
        check_experiment_ids,
        check_results_schema,
        check_tests,
    ]

    all_passed = True
    for check in checks:
        if not check():
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("ALL GUARDS PASSED: Project environment is verified and ready.")
        return 0
    else:
        print("GUARD FAILURE: Please fix the issues above before running experiments.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
