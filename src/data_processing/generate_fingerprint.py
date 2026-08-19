"""Generate and verify dataset fingerprint for the master CFPB dataset."""

from pathlib import Path
import sys

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_raw
from src.fingerprint import create_dataset_fingerprint, verify_dataset_fingerprint

if __name__ == "__main__":
    parquet_path = Path("data/combined_complaints.parquet")
    manifest_path = Path("data/dataset_manifest.json")

    print(f"Loading dataset from {parquet_path}...")
    df = load_raw(parquet_path)
    print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns.")

    print("Generating dataset fingerprint...")
    fp = create_dataset_fingerprint(df_or_path=parquet_path)

    print("\n=== Dataset Fingerprint Summary ===")
    print(f"File Path:                  {fp.file_path}")
    print(f"File Size:                  {fp.file_size_bytes:,} bytes")
    print(f"File SHA-256:               {fp.file_sha256}")
    print(f"Deterministic Content SHA:  {fp.content_sha256}")
    print(f"Total Rows:                 {fp.total_rows:,}")
    print(f"Unique Narratives:          {fp.unique_narratives:,}")
    print(f"Duplicate Narratives:       {fp.duplicate_narratives_count:,}")
    print(f"Class Distribution:         {fp.class_distribution}")
    print(f"Class Ratios:               {fp.class_ratios}")

    print(f"\nSaving manifest to {manifest_path}...")
    fp.save(manifest_path)
    print("Manifest saved successfully.")

    print("\nVerifying dataset against saved manifest...")
    is_valid, mismatches = verify_dataset_fingerprint(parquet_path, manifest_path)
    if is_valid:
        print("Dataset verification PASSED with 0 mismatches.")
    else:
        print(f"Dataset verification FAILED: {mismatches}")
