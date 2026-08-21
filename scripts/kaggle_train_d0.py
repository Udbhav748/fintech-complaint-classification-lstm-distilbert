"""D0 training kernel, run on Kaggle GPU. Templated: {{SEED}} is substituted by
scripts/run_d0.py before each `kaggle kernels push` - this file is never
pushed as-is, and is not meant to be run locally.

Differs from scripts/kaggle_train_m*.py in kind, not just config: D0 is a
pretrained Transformer, not the recurrent factory, so this script builds via
`src.distilbert` (never `src.models.build_recurrent_model`), tokenizes with
the pretrained DistilBERT tokenizer (never the frozen Keras tokenizer, never
fit here), and needs internet access - the only project kernel that does -
to pull the pretrained checkpoint from the HuggingFace Hub and to pin
`transformers`/`tf_keras` versions, since Kaggle's preinstalled `transformers`
may also be the TF-incompatible 5.x series (see requirements.txt).
"""

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

INPUT_DIR = Path("/kaggle/input/datasets/udbhav748/cfpb-d0-frozen-inputs")
OUTPUT_DIR = Path("/kaggle/working")
SEED = {{SEED}}
EXPERIMENT = "D0"

# Pinned before anything else imports transformers - transformers>=5.0 has no
# TF model classes at all; tf_keras is required for the TF classes to import
# under TF's native Keras 3. Matches requirements.txt exactly.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "transformers==4.57.6", "tf_keras>=2.15.0"],
    check=True,
)

sys.path.insert(0, str(INPUT_DIR))

import numpy as np

from src.checkpoint import get_checkpoint_path, save_checkpoint_record
from src.config import get_experiment_config
from src.data import LABEL_COL, LABELS, TEXT_COL, deduplicate, load_raw
from src.distilbert import build_d0_callbacks, build_d0_model, compile_d0_model, verify_d0_config
from src.evaluation import compute_metrics, compute_val_macro_f1
from src.fingerprint import SplitFingerprint, verify_dataset_fingerprint, verify_split_fingerprint
from src.models import count_parameters
from src.preprocessing import class_ids, tokenize_for_distilbert
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

    # 3. D0 config - checkpoint, max_len, and every hyperparameter come from
    # here, never invented in this script.
    cfg = get_experiment_config(EXPERIMENT)
    checkpoint, max_len = cfg["pretrained_model_name"], cfg["max_len"]
    verify_d0_config(checkpoint, max_len, num_labels=len(LABELS))
    print(f"D0 config verified: checkpoint={checkpoint} max_len={max_len}")

    # 4. Pretrained DistilBERT tokenizer - never fit on this dataset. Same
    # tokenize_for_distilbert function Task 3 already built and tested.
    def to_arrays(idx):
        encoded = tokenize_for_distilbert(df.iloc[idx][TEXT_COL], max_len=max_len, checkpoint=checkpoint)
        y = class_ids(df.iloc[idx][LABEL_COL])
        return encoded["input_ids"], encoded["attention_mask"], y

    X_train_ids, X_train_mask, y_train = to_arrays(train_idx)
    X_val_ids, X_val_mask, y_val = to_arrays(val_idx)
    X_test_ids, X_test_mask, y_test = to_arrays(test_idx)
    print(f"input shapes: train={X_train_ids.shape} val={X_val_ids.shape} test={X_test_ids.shape}")

    trunc_256 = float(np.mean(X_train_mask.sum(axis=1) >= max_len)) * 100
    print(f"train sequences using the full {max_len}-token window (proxy for truncation): {trunc_256:.1f}%")

    # 5. Build + compile D0. `set_seed` before model construction, matching
    # M0-M4's convention, for the classification head's random init.
    set_seed(SEED)
    model = build_d0_model(checkpoint, num_labels=len(LABELS))
    model = compile_d0_model(model, learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    params = count_parameters(model)
    print(f"Parameters: {params}")

    # 6. Callbacks - ValMacroF1Callback (recomputes val_macro_f1 each epoch via
    # compute_val_macro_f1) + ModelCheckpoint(val_macro_f1) +
    # EarlyStopping(patience=2), all from src.distilbert.build_d0_callbacks,
    # config-driven.
    checkpoint_path = get_checkpoint_path(EXPERIMENT, seed=SEED, checkpoint_dir=OUTPUT_DIR / "checkpoints")
    callbacks = build_d0_callbacks(checkpoint_path, X_val_ids, X_val_mask, y_val)
    print(f"Callbacks: {[c.__class__.__name__ for c in callbacks]}")

    # 7. Train. epochs=4 is the locked ceiling from config - never raised here.
    history = model.fit(
        x={"input_ids": X_train_ids, "attention_mask": X_train_mask},
        y=y_train,
        validation_data=({"input_ids": X_val_ids, "attention_mask": X_val_mask}, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )
    train_time = time.time() - run_start

    # 8. Explicit restore from the ModelCheckpoint file - same code path as
    # M0-M4, independent of whether EarlyStopping's own restore fired.
    model.load_weights(str(checkpoint_path))

    # 9. Cross-check the best-epoch val macro-F1 two independent ways.
    val_f1_history = history.history["val_macro_f1"]
    best_epoch = int(np.argmax(val_f1_history)) + 1
    best_val_f1_from_history = float(max(val_f1_history))

    val_logits = model.predict({"input_ids": X_val_ids, "attention_mask": X_val_mask}, verbose=0).logits
    val_pred = np.argmax(val_logits, axis=-1)
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
    test_logits = model.predict({"input_ids": X_test_ids, "attention_mask": X_test_mask}, verbose=0).logits
    test_pred = np.argmax(test_logits, axis=-1)
    test_metrics = compute_metrics(y_test, test_pred, labels=LABELS)

    # 11. Persist everything needed to trace and validate this run.
    OUTPUT_DIR.mkdir(exist_ok=True)

    save_checkpoint_record(
        experiment=EXPERIMENT, seed=SEED, best_epoch=best_epoch,
        best_val_macro_f1=best_val_f1_independent, checkpoint_file=checkpoint_path,
        model_architecture={"checkpoint": checkpoint, "max_len": max_len, "num_labels": len(LABELS)},
        checkpoint_dir=OUTPUT_DIR / "checkpoints",
    )

    split_fp = SplitFingerprint.load(split_manifest)

    run_record = {
        "experiment": EXPERIMENT, "model": "DistilBERT (fine-tuned transformer)",
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

    (OUTPUT_DIR / "truncation.json").write_text(json.dumps({
        "max_len": max_len, "train_sequences_using_full_window_%": round(trunc_256, 2),
    }, indent=2), encoding="utf-8")

    import transformers
    (OUTPUT_DIR / "tokenizer_info.json").write_text(json.dumps({
        "checkpoint": checkpoint,
        "tokenizer_class": "DistilBertTokenizerFast",
        "transformers_version": transformers.__version__,
    }, indent=2), encoding="utf-8")

    (OUTPUT_DIR / "environment.json").write_text(
        json.dumps(EnvironmentInfo.capture().to_dict(), indent=2), encoding="utf-8"
    )

    print("Run complete. Outputs written to /kaggle/working.")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
