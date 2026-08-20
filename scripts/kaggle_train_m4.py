"""M4 training kernel, run on Kaggle GPU. Templated: {{SEED}} is substituted by
scripts/run_m4.py before each `kaggle kernels push` - this file is never pushed
as-is, and is not meant to be run locally (no local GPU training of M4, per
project policy).

Identical pipeline to scripts/kaggle_train_m3.py. The intended differences
between M3 and M4 are exactly what src.config.EXPERIMENT_CONFIGS["M4"] bundles:
max_len=256, ReduceLROnPlateau, and EarlyStopping - this script does not branch
on experiment identity anywhere beyond reading that config. Per the project
plan's own limitation (project_plan.md Stage 10), these three changes are
bundled and their individual contributions are not separately identified by
this run - only the combined M4-vs-M3 effect is scientifically valid here.

One addition not present in scripts/kaggle_train_m3.py: a small callback that
records the optimizer's learning rate at the end of every epoch, since
ReduceLROnPlateau does not add "lr" to Keras's history.history by itself, and
Part 11 of the task asks for LR history/reduction events to be recorded.
"""

from dataclasses import asdict
import json
import sys
import time
from pathlib import Path

INPUT_DIR = Path("/kaggle/input/datasets/udbhav748/cfpb-m4-frozen-inputs")
OUTPUT_DIR = Path("/kaggle/working")
SEED = {{SEED}}
EXPERIMENT = "M4"

sys.path.insert(0, str(INPUT_DIR))

import keras
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


class LearningRateLogger(keras.callbacks.Callback):
    """Records the optimizer's learning rate at the end of every epoch, so
    ReduceLROnPlateau's reduction events are visible after training - Keras
    does not put this in history.history on its own."""

    def __init__(self):
        super().__init__()
        self.lr_history: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        self.lr_history.append(float(lr.numpy() if hasattr(lr, "numpy") else lr))


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

    # 3. Frozen tokenizer (never refit) and sequences at M4's locked max_len=256 -
    # the one input-shape change from M3. Same tokenizer, same vocabulary.
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
    print(f"input shape: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    # 3b. Truncation statistics at max_len=256, compared against Task 3's audit.
    train_lens = [len(s) for s in tokenizer.texts_to_sequences(df.iloc[train_idx][TEXT_COL].map(normalize_text))]
    trunc_256 = sum(1 for l in train_lens if l > max_len) / len(train_lens) * 100
    trunc_128 = sum(1 for l in train_lens if l > 128) / len(train_lens) * 100
    print(f"train truncation: {trunc_128:.1f}% at 128, {trunc_256:.1f}% at {max_len}")

    # 3c. Frozen GloVe matrix (Task 3), loaded as-is - the same matrix M3 used.
    embedding_matrix = np.load(INPUT_DIR / "artifacts" / "embeddings" / "glove_100d_matrix.npy")
    expected_shape = (cfg["vocab_size"], cfg["embedding_dim"])
    if embedding_matrix.shape != expected_shape:
        raise RuntimeError(f"GloVe matrix shape {embedding_matrix.shape} != expected {expected_shape}")
    print(f"GloVe matrix loaded: shape={embedding_matrix.shape}")

    # 4. Build M4 - GloVe embeddings, max_len=256. The only architecture
    # difference from M3 is the input shape; dropout/recurrent_dropout/
    # spatial_dropout are unchanged (read from config, not hardcoded here).
    set_seed(SEED)
    model_config = RecurrentModelConfig.from_experiment(EXPERIMENT)
    model = build_recurrent_model(model_config, embedding_matrix=embedding_matrix)

    # 5. Architecture visible in the log without downloading anything.
    print(f"Architecture: {[l.__class__.__name__ for l in model.layers]}")
    print(f"Parameters: {count_parameters(model)}")

    # 6. Callbacks - for M4 this is
    # [ModelCheckpoint, ReduceLROnPlateau, EarlyStopping], all monitoring
    # val_macro_f1, built entirely from config by the factory (Part 6). The
    # LR logger is added here, not in the factory, since it is purely
    # observational and not part of the locked training policy.
    checkpoint_path = get_checkpoint_path(EXPERIMENT, seed=SEED, checkpoint_dir=OUTPUT_DIR / "checkpoints")
    callbacks = build_callbacks(model_config, checkpoint_path)
    lr_logger = LearningRateLogger()
    callbacks.append(lr_logger)
    print(f"Callbacks: {[c.__class__.__name__ for c in callbacks]}")

    # 7. Train. epochs=10 is the locked ceiling from config (project_plan.md
    # §4/§13: max 10, ladder-wide) - never raised here. EarlyStopping may stop
    # training before 10 epochs and restores the best validation weights when
    # it does; that is a termination behavior, not a separate accuracy lever.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )
    train_time = time.time() - run_start

    # 8. EarlyStopping(restore_best_weights=True) already restores the best
    # weights into `model` when it fires. Loading from the checkpoint file
    # explicitly as well keeps this identical to every prior rung's code path
    # (ModelCheckpoint's file is the single source of truth for "best"), and is
    # a no-op if EarlyStopping already restored the same weights.
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
        "experiment": EXPERIMENT, "model": "BiLSTM + GloVe + LR schedule + EarlyStopping + max_len=256",
        "seed": SEED, "macro_f1": test_metrics["macro_f1"], "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"], "macro_recall": test_metrics["macro_recall"],
        "best_epoch": best_epoch, "epochs_run": len(history.epoch),
        "train_time": train_time, "parameter_count": params["total"],
        "checkpoint_path": str(checkpoint_path), "dataset_version": split_fp.dataset_content_sha256,
    }
    (OUTPUT_DIR / "run_record.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    (OUTPUT_DIR / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    (OUTPUT_DIR / "lr_history.json").write_text(json.dumps({
        "lr_by_epoch": lr_logger.lr_history,
        "base_learning_rate": cfg["learning_rate"],
        "reductions": [
            {"epoch": i + 1, "from": lr_logger.lr_history[i - 1], "to": lr_logger.lr_history[i]}
            for i in range(1, len(lr_logger.lr_history))
            if lr_logger.lr_history[i] < lr_logger.lr_history[i - 1]
        ],
    }, indent=2), encoding="utf-8")

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
        "max_len": max_len, "truncated_at_128_%": round(trunc_128, 2), "truncated_at_256_%": round(trunc_256, 2),
    }, indent=2), encoding="utf-8")

    (OUTPUT_DIR / "environment.json").write_text(
        json.dumps(EnvironmentInfo.capture().to_dict(), indent=2), encoding="utf-8"
    )

    print("Run complete. Outputs written to /kaggle/working.")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
