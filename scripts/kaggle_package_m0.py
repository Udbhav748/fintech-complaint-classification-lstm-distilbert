"""Builds and uploads the Kaggle Dataset the M0 training kernel reads from.

Packages exactly what src/models.py + src/preprocessing.py need to build and
train M0 on Kaggle: the frozen parquet + manifests, the frozen split indices, the
frozen (train-fitted) tokenizer, data_config.json, and a copy of src/ so the
kernel can `import src.*` unmodified. No GloVe matrix - M0 uses random
embeddings, and src.models.build_recurrent_model already refuses a matrix for a
"random" embedding_type, so leaving GloVe out is also a cheap extra guarantee.

Verifies the local artifacts against their own manifests before packaging, so a
stale or corrupted local file can't silently get shipped to Kaggle as if it were
the frozen one.

    python scripts/kaggle_package_m0.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import deduplicate, load_raw
from src.fingerprint import verify_dataset_fingerprint, verify_split_fingerprint
from src.split import load_splits

DATASET_SLUG = "cfpb-m0-frozen-inputs"
STAGING_DIR = REPO_ROOT / "kaggle_staging" / DATASET_SLUG

FILES_TO_COPY = [
    ("data/combined_complaints.parquet", "data/combined_complaints.parquet"),
    ("data/dataset_manifest.json", "data/dataset_manifest.json"),
    ("data/splits/train_idx.npy", "data/splits/train_idx.npy"),
    ("data/splits/val_idx.npy", "data/splits/val_idx.npy"),
    ("data/splits/test_idx.npy", "data/splits/test_idx.npy"),
    ("data/splits/split_manifest.json", "data/splits/split_manifest.json"),
    ("artifacts/tokenizer/keras_tokenizer.json", "artifacts/tokenizer/keras_tokenizer.json"),
    ("configs/data_config.json", "configs/data_config.json"),
]


def verify_local_inputs() -> None:
    """Same check the Kaggle kernel repeats on its own copy - done here first so
    a bad local file is caught before spending an upload on it."""
    df_raw = load_raw()
    verify_dataset_fingerprint(df_raw, REPO_ROOT / "data/dataset_manifest.json")
    df = deduplicate(df_raw)
    train_idx, val_idx, test_idx = load_splits(REPO_ROOT / "data/splits")
    verify_split_fingerprint(df, train_idx, val_idx, test_idx, REPO_ROOT / "data/splits/split_manifest.json")
    print("Local dataset + split fingerprints verified.")


def stage_files() -> None:
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)

    for src_rel, dst_rel in FILES_TO_COPY:
        src = REPO_ROOT / src_rel
        dst = STAGING_DIR / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    src_pkg = STAGING_DIR / "src"
    shutil.copytree(
        REPO_ROOT / "src", src_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data_processing"),
    )

    metadata = {
        "title": "CFPB M0 frozen inputs",
        "id": f"udbhav748/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (STAGING_DIR / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    total_bytes = sum(f.stat().st_size for f in STAGING_DIR.rglob("*") if f.is_file())
    print(f"Staged {total_bytes / 1e6:.1f} MB at {STAGING_DIR}")


def dataset_exists() -> bool:
    result = subprocess.run(
        ["kaggle", "datasets", "status", f"udbhav748/{DATASET_SLUG}"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def upload() -> str:
    """Creates the dataset on first run, versions it on subsequent runs. Returns
    the version number as a string (parsed from the CLI's own confirmation), so
    the orchestrator can stamp it onto each seed's traceability record."""
    if dataset_exists():
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(STAGING_DIR), "-m", "M0 training run", "-r", "zip"],
            capture_output=True, text=True, cwd=str(STAGING_DIR),
        )
    else:
        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(STAGING_DIR), "-r", "zip"],
            capture_output=True, text=True, cwd=str(STAGING_DIR),
        )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"kaggle datasets upload failed (exit {result.returncode})")
    return result.stdout


if __name__ == "__main__":
    verify_local_inputs()
    stage_files()
    upload()
