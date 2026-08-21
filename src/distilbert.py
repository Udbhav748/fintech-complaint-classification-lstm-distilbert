"""D0 model and tokenizer construction - DistilBERT fine-tuning.

Deliberately separate from `src.models` (the M0-M4 recurrent factory): D0's
architecture is a pretrained Transformer, not an LSTM, and project_plan.md's
own Task 13 brief requires D0 not import `build_recurrent_model`/
`RecurrentModelConfig`. `count_parameters` IS reused from `src.models` - it is
architecture-generic (a pure function summing `model.trainable_weights`).

`MacroF1Score` (`src.models`) is NOT reused here, for a concrete compatibility
reason, not a style choice: HF's TF model classes are built on `tf_keras`
(the legacy Keras 2 compatibility shim), while `src.models` and its
`MacroF1Score` use TF's native Keras 3 (`import keras`). Compiling a
`tf_keras`-based model with a Keras-3 metric/optimizer/loss raises
`AttributeError: 'Variable' object has no attribute '_distribute_strategy'`
internally - confirmed by hitting exactly that error before adopting this
design. Rather than subclass `tf_keras.metrics.Metric` with a second
confusion-matrix accumulator (a real duplicate implementation), D0's
val_macro_f1 is produced by a callback that calls
`src.evaluation.compute_val_macro_f1` directly on `on_epoch_end` and writes
it into `logs` - the *same* tested function every other rung's final metric
comes from, not a reimplementation, and simpler than accumulating a
confusion matrix inside a TF `Metric` object. Everything that touches the D0
model's `.compile()`/`.fit()` (optimizer, loss, callbacks) therefore comes
from `tf_keras`, not the top-level `keras` package.

Requires TensorFlow and `transformers`. `transformers` must be pinned below
5.0 (see requirements.txt) - the 5.x series removed TensorFlow model classes
(`TFAutoModelForSequenceClassification` and friends) entirely, and `tf_keras`
is required alongside TF's native Keras 3 for the TF model classes to import
at all. `use_safetensors=False` is required when loading
`distilbert-base-uncased`: this transformers version's PyTorch-safetensors-
to-TF conversion path is broken (`'builtins.safe_open' object is not
iterable`), but the checkpoint also ships native `tf_model.h5` weights, which
`use_safetensors=False` selects directly - verified working locally before
this was adopted.
"""

from pathlib import Path
from typing import Union

import numpy as np
import tf_keras
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, PreTrainedTokenizerFast

from src.config import get_experiment_config
from src.data import LABELS
from src.evaluation import compute_val_macro_f1
from src.models import count_parameters

D0_EXPERIMENT = "D0"


class ValMacroF1Callback(tf_keras.callbacks.Callback):
    """Computes val_macro_f1 via `src.evaluation.compute_val_macro_f1` (the
    project's one canonical implementation, already verified against sklearn)
    on the full validation set at the end of every epoch, and writes it into
    `logs` so `ModelCheckpoint`/`EarlyStopping` can monitor it exactly as they
    do for M0-M4 - just produced by a direct recompute here instead of a
    `tf_keras`-compatible Metric class."""

    def __init__(self, val_input_ids: np.ndarray, val_attention_mask: np.ndarray, val_labels: np.ndarray):
        super().__init__()
        self.val_input_ids = val_input_ids
        self.val_attention_mask = val_attention_mask
        self.val_labels = val_labels

    def on_epoch_end(self, epoch, logs=None):
        logits = self.model.predict(
            {"input_ids": self.val_input_ids, "attention_mask": self.val_attention_mask},
            verbose=0,
        ).logits
        val_pred = np.argmax(logits, axis=-1)
        if logs is None:
            logs = {}
        logs["val_macro_f1"] = compute_val_macro_f1(self.val_labels, val_pred, labels=LABELS)


def build_d0_tokenizer(checkpoint: str) -> PreTrainedTokenizerFast:
    """Pretrained DistilBERT tokenizer, never fit on this dataset - same
    checkpoint family as the model, never substituted."""
    return AutoTokenizer.from_pretrained(checkpoint)


def build_d0_model(checkpoint: str, num_labels: int = len(LABELS)) -> tf_keras.Model:
    """Pretrained DistilBERT + a fresh classification head, loaded from the
    checkpoint's native TF weights (`use_safetensors=False` - see module
    docstring). `num_labels` defaults to `len(LABELS)` (5), read from
    `src.data`, not hardcoded - matches every other output-size check in the
    project (`src.models.RecurrentModelConfig.__post_init__`)."""
    return TFAutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels, use_safetensors=False
    )


def compile_d0_model(model: tf_keras.Model, learning_rate: float, weight_decay: float) -> tf_keras.Model:
    """AdamW + sparse categorical crossentropy from logits (HF TF sequence
    classification heads return raw logits, not softmax probabilities).
    `accuracy` is the only compiled metric - val_macro_f1 comes from
    `ValMacroF1Callback`, not a compiled metric (see module docstring)."""
    optimizer = tf_keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(
        optimizer=optimizer,
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_d0_callbacks(
    checkpoint_path: Union[str, Path],
    val_input_ids: np.ndarray,
    val_attention_mask: np.ndarray,
    val_labels: np.ndarray,
) -> list[tf_keras.callbacks.Callback]:
    """`ValMacroF1Callback` first (it must populate `logs["val_macro_f1"]`
    before the callbacks that read it run - Keras calls callbacks in list
    order), then `ModelCheckpoint(monitor="val_macro_f1", save_best_only=True)`
    - same fixed policy as M0-M4 - then `EarlyStopping(patience=2)`, D0's own
    locked value (`EXPERIMENT_CONFIGS["D0"]["early_stopping_patience"]`).
    D0's config has no `lr_schedule` key, so no scheduler callback is added -
    not omitted, never configured for D0 in the first place."""
    cfg = get_experiment_config(D0_EXPERIMENT)
    callbacks: list[tf_keras.callbacks.Callback] = [
        ValMacroF1Callback(val_input_ids, val_attention_mask, val_labels),
        tf_keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]
    if cfg.get("early_stopping"):
        callbacks.append(
            tf_keras.callbacks.EarlyStopping(
                monitor="val_macro_f1",
                mode="max",
                patience=cfg["early_stopping_patience"],
            )
        )
    return callbacks


def verify_d0_config(checkpoint: str, max_len: int, num_labels: int) -> None:
    """Part 2's compatibility checks, done once before training rather than
    discovered mid-run. Raises loudly rather than silently substituting
    anything."""
    cfg = get_experiment_config(D0_EXPERIMENT)
    if cfg["pretrained_model_name"] != checkpoint:
        raise ValueError(
            f"Checkpoint mismatch: config says '{cfg['pretrained_model_name']}', got '{checkpoint}'"
        )
    if cfg["max_len"] != max_len:
        raise ValueError(f"max_len mismatch: config says {cfg['max_len']}, got {max_len}")
    if num_labels != len(LABELS):
        raise ValueError(f"num_labels must equal len(LABELS)={len(LABELS)}, got {num_labels}")

    tokenizer = build_d0_tokenizer(checkpoint)
    required_special = {"cls_token", "sep_token", "pad_token"}
    missing = required_special - set(tokenizer.special_tokens_map)
    if missing:
        raise ValueError(f"Tokenizer for '{checkpoint}' is missing required special tokens: {missing}")

    encoded = tokenizer(["compatibility check"], truncation=True, max_length=max_len,
                         padding="max_length", return_tensors="np")
    if encoded["input_ids"].shape != (1, max_len):
        raise ValueError(f"input_ids shape {encoded['input_ids'].shape} != (1, {max_len})")
    if encoded["attention_mask"].shape != (1, max_len):
        raise ValueError(f"attention_mask shape {encoded['attention_mask'].shape} != (1, {max_len})")
