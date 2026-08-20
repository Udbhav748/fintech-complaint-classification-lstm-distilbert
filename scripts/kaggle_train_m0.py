"""M0 training kernel, run on Kaggle GPU. Templated: {{SEED}} is substituted by
scripts/run_m0.py before each `kaggle kernels push` - this file is never pushed
as-is, and is not meant to be run locally (no local GPU training of M0, per
project policy).

Order matches project_plan.md Task 6 Part 3 exactly: verify -> load -> build ->
train (val-monitored only) -> restore best -> evaluate test (only now) -> save
everything needed to trace this run back to the exact frozen inputs that produced
it.
"""

from dataclasses import asdict
import json
import sys
import time
from pathlib import Path

INPUT_DIR = Path("/kaggle/input/datasets/udbhav748/cfpb-m0-frozen-inputs")
OUTPUT_DIR = Path("/kaggle/working")
SEED = {{SEED}}
EXPERIMENT = "M0"

sys.path.insert(0, str(INPUT_DIR))

import numpy as np

from src.checkpoint import get_checkpoint_path, save_checkpoint_record
from src.config import get_experiment_config
from src.data import LABEL_COL, LABELS, TEXT_COL, deduplicate, load_raw
from src.evaluation import compute_metrics, compute_val_macro_f1
from src.fingerprint import SplitFingerprint, verify_dataset_fingerprint, verify_split_fingerprint
from src.keras_tokenizer import KerasTokenizer, pad_sequences
from src.models import RecurrentModelConfig, build_callbacks, build_recurrent_model, count_parameters
from src.preprocessing import class_ids, normalize_text
from src.reproducibility import EnvironmentInfo, set_seed
from src.split import load_splits


def main() -> None:
    run_start = time.time()

    # 1-2. Load and verify the Kaggle copy is really what Task 1/2 froze.
    df_raw = load_raw(INPUT_DIR / "data" / "combined_complaints.parquet")
    verify_dataset_fingerprint(df_raw, INPUT_DIR / "data" / "dataset_manifest.json")
    df = deduplicate(df_raw)
    split_manifest = INPUT_DIR / "data" / "splits" / "split_manifest.json"
    train_idx, val_idx, test_idx = load_splits(INPUT_DIR / "data" / "splits")
    verify_split_fingerprint(df, train_idx, val_idx, test_idx, split_manifest)
    print(f"Verified: {len(df)} rows, train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # 3. Frozen tokenizer (never refit) and sequences at M0's locked max_len=128.
    tokenizer = KerasTokenizer.load(INPUT_DIR / "artifacts" / "tokenizer" / "keras_tokenizer.json")
    cfg = get_experiment_config(EXPERIMENT)
    max_len = cfg["max_len"]

    def to_arrays(idx):
        normalized = df.iloc[idx][TEXT_COL].map(normalize_text)
        sequences = tokenizer.texts_to_sequences(normalized)
        X = pad_sequences(sequences, max_len=max_len, padding="pre", truncating="post")
        y = class_ids(df.iloc[idx][LABEL_COL])
        return X, y

    X_train, y_train = to_arrays(train_idx)
    X_val, y_val = to_arrays(val_idx)
    X_test, y_test = to_arrays(test_idx)

    # 4. Build M0 - random embeddings, no matrix passed.
    set_seed(SEED)
    model_config = RecurrentModelConfig.from_experiment(EXPERIMENT)
    model = build_recurrent_model(model_config)

    # 5. Architecture visible in the log without downloading anything.
    print(f"Architecture: {[l.__class__.__name__ for l in model.layers]}")
    print(f"Parameters: {count_parameters(model)}")

    # 6. Callbacks - for M0 this is exactly [ModelCheckpoint(val_macro_f1)],
    # enforced by the factory itself (no scheduler/early-stopping fields are set
    # in M0's config, so build_callbacks adds neither).
    checkpoint_path = get_checkpoint_path(EXPERIMENT, seed=SEED, checkpoint_dir=OUTPUT_DIR / "checkpoints")
    callbacks = build_callbacks(model_config, checkpoint_path)
    print(f"Callbacks: {[c.__class__.__name__ for c in callbacks]}")

    # 7. Train. epochs=10 is the locked ceiling from config - never raised here.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )
    train_time = time.time() - run_start

    # 8. ModelCheckpoint only writes to disk; it does not restore into `model`.
    # Without this, "best" and "final-epoch" would silently differ.
    model.load_weights(str(checkpoint_path))

    # 9. Cross-check the best-epoch val macro-F1 two independent ways.
    val_f1_history = history.history["val_macro_f1"]
    best_epoch = int(np.argmax(val_f1_history)) + 1
    best_val_f1_from_history = float(max(val_f1_history))

    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=-1)
    best_val_f1_independent = compute_val_macro_f1(y_val, val_pred, labels=LABELS)

    agreement = abs(best_val_f1_from_history - best_val_f1_independent)
    print(f"val_macro_f1 (Keras history, best epoch): {best_val_f1_from_history:.6f}")
    print(f"val_macro_f1 (independent sklearn recompute, restored weights): {best_val_f1_independent:.6f}")
    print(f"agreement: {agreement:.6f}")
    if agreement > 1e-3:
        raise RuntimeError(
            f"val_macro_f1 mismatch between Keras history and independent recompute "
            f"({best_val_f1_from_history} vs {best_val_f1_independent}) - checkpoint "
            f"selection cannot be trusted; investigate before saving results."
        )

    # 10. Test set only now, after checkpoint selection is finalized.
    test_pred = np.argmax(model.predict(X_test, verbose=0), axis=-1)
    test_metrics = compute_metrics(y_test, test_pred, labels=LABELS)

    # 11. Persist everything needed to trace and validate this run.
    OUTPUT_DIR.mkdir(exist_ok=True)
    params = count_parameters(model)

    save_checkpoint_record(
        experiment=EXPERIMENT, seed=SEED, best_epoch=best_epoch,
        best_val_macro_f1=best_val_f1_independent, checkpoint_file=checkpoint_path,
        model_architecture={"layers": [l.__class__.__name__ for l in model.layers], **asdict(model_config)},
        checkpoint_dir=OUTPUT_DIR / "checkpoints",
    )

    # dataset_version comes from the verified split manifest, not guessed.
    split_fp = SplitFingerprint.load(split_manifest)

    run_record = {
        "experiment": EXPERIMENT, "model": "Unidirectional LSTM baseline, random embeddings",
        "seed": SEED, "macro_f1": test_metrics["macro_f1"], "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"], "macro_recall": test_metrics["macro_recall"],
        "best_epoch": best_epoch, "epochs_run": len(history.epoch),
        "train_time": train_time, "parameter_count": params["total"],
        "checkpoint_path": str(checkpoint_path), "dataset_version": split_fp.dataset_content_sha256,
    }
    (OUTPUT_DIR / "run_record.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    (OUTPUT_DIR / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    (OUTPUT_DIR / "val_check.json").write_text(json.dumps({
        "best_val_macro_f1_from_history": best_val_f1_from_history,
        "best_val_macro_f1_independent_sklearn": best_val_f1_independent,
        "agreement": agreement,
    }, indent=2), encoding="utf-8")

    np.savetxt(
        OUTPUT_DIR / "test_predictions.csv",
        np.column_stack([test_idx, y_test, test_pred]),
        delimiter=",", header="test_idx,y_true,y_pred", comments="", fmt="%d",
    )

    (OUTPUT_DIR / "test_metrics.json").write_text(json.dumps({
        "macro_f1": test_metrics["macro_f1"], "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"], "macro_recall": test_metrics["macro_recall"],
        "per_class": test_metrics["per_class"], "confusion_matrix": test_metrics["confusion_matrix"],
        "labels": LABELS,
    }, indent=2), encoding="utf-8")

    (OUTPUT_DIR / "environment.json").write_text(
        json.dumps(EnvironmentInfo.capture().to_dict(), indent=2), encoding="utf-8"
    )

    print("Run complete. Outputs written to /kaggle/working.")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
