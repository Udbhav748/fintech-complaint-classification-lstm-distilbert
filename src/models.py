"""Recurrent model factory for M0-M4.

Builds one architecture from one config source (`src.config.EXPERIMENT_CONFIGS`),
with each rung changing exactly the fields the experiment ladder says it changes.
D0 is DistilBERT and is not built here.

Requires TensorFlow/Keras. Nothing else in `src/` does - `src.evaluation` and
`src.preprocessing` are deliberately framework-agnostic so the rest of the test
suite runs without it. This module is the one place that boundary is crossed,
because building an LSTM requires an actual deep learning framework; there is no
honest way to reimplement that in pure Python the way `src.keras_tokenizer` could
reimplement word-index construction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import keras
import numpy as np
import tensorflow as tf

from src.config import get_experiment_config
from src.data import LABELS

RECURRENT_EXPERIMENTS = ["M0", "M1", "M2", "M3", "M4"]


class ModelConfigError(ValueError):
    """Raised when a model configuration is invalid or a required hyperparameter
    is missing, rather than letting Keras fail several layers deeper."""
    pass


@dataclass
class RecurrentModelConfig:
    """A typed, validated view over one experiment's entry in
    `src.config.EXPERIMENT_CONFIGS` - not a second set of constants. Every field
    is read from that dict; nothing here is invented.

    Construct via `RecurrentModelConfig.from_experiment("M2")` for the real,
    locked M0-M4 settings. The bare constructor is for tests only, where Part 19
    explicitly calls for tiny synthetic dimensions rather than the real 20k-word,
    128-unit configuration.
    """
    experiment_id: str
    vocab_size: int
    embedding_dim: int
    embedding_type: str  # "random" | "glove"
    embedding_trainable: bool
    max_len: int
    lstm_units: int
    bidirectional: bool
    dropout: float
    recurrent_dropout: float
    spatial_dropout: float
    output_classes: int
    learning_rate: float
    optimizer: str
    lr_schedule: Optional[str] = None
    lr_schedule_factor: Optional[float] = None
    lr_schedule_patience: Optional[int] = None
    lr_schedule_min_lr: Optional[float] = None
    early_stopping: bool = False
    early_stopping_patience: Optional[int] = None
    early_stopping_restore_best_weights: bool = True

    def __post_init__(self) -> None:
        errors = []
        if self.experiment_id not in RECURRENT_EXPERIMENTS:
            errors.append(
                f"experiment_id '{self.experiment_id}' must be one of {RECURRENT_EXPERIMENTS} "
                f"(D0 is DistilBERT, not built by this factory)"
            )
        if self.vocab_size <= 0:
            errors.append(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.embedding_dim <= 0:
            errors.append(f"embedding_dim must be > 0, got {self.embedding_dim}")
        if self.max_len <= 0:
            errors.append(f"max_len must be > 0, got {self.max_len}")
        if self.lstm_units <= 0:
            errors.append(f"lstm_units must be > 0, got {self.lstm_units}")
        if self.output_classes != len(LABELS):
            errors.append(f"output_classes must equal {len(LABELS)} (len(LABELS)), got {self.output_classes}")
        if self.embedding_type not in ("random", "glove"):
            errors.append(f"embedding_type must be 'random' or 'glove', got '{self.embedding_type}'")
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be > 0, got {self.learning_rate}")
        for name, val in [("dropout", self.dropout), ("recurrent_dropout", self.recurrent_dropout),
                          ("spatial_dropout", self.spatial_dropout)]:
            if not (0.0 <= val < 1.0):
                errors.append(f"{name} must be in [0, 1), got {val}")
        if errors:
            raise ModelConfigError("Invalid RecurrentModelConfig:\n" + "\n".join(f"- {e}" for e in errors))

    @classmethod
    def from_experiment(cls, experiment_id: str) -> "RecurrentModelConfig":
        """Reads `src.config.EXPERIMENT_CONFIGS[experiment_id]` - the single
        source of truth for M0-M4 hyperparameters - into this typed structure."""
        if experiment_id not in RECURRENT_EXPERIMENTS:
            raise ModelConfigError(
                f"'{experiment_id}' is not a recurrent experiment. Must be one of "
                f"{RECURRENT_EXPERIMENTS} (D0 is DistilBERT, built separately)."
            )
        cfg = get_experiment_config(experiment_id)
        return cls(
            experiment_id=experiment_id,
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["embedding_dim"],
            embedding_type="glove" if cfg["use_glove"] else "random",
            embedding_trainable=cfg["glove_trainable"],
            max_len=cfg["max_len"],
            lstm_units=cfg["hidden_dim"],
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            recurrent_dropout=cfg.get("recurrent_dropout", 0.0),
            spatial_dropout=cfg.get("spatial_dropout", 0.0),
            output_classes=len(LABELS),
            learning_rate=cfg["learning_rate"],
            optimizer=cfg["optimizer"],
            lr_schedule=cfg.get("lr_schedule"),
            lr_schedule_factor=cfg.get("lr_schedule_factor"),
            lr_schedule_patience=cfg.get("lr_schedule_patience"),
            lr_schedule_min_lr=cfg.get("lr_schedule_min_lr"),
            early_stopping=cfg.get("early_stopping", False),
            early_stopping_patience=cfg.get("early_stopping_patience"),
            early_stopping_restore_best_weights=cfg.get("early_stopping_restore_best_weights", True),
        )


class MacroF1Score(keras.metrics.Metric):
    """Macro-F1 as a genuine Keras metric - not a per-batch average.

    Accumulates a confusion matrix across every batch in an epoch (`update_state`
    is called once per batch by `model.fit`/`model.evaluate`; `reset_state` runs
    at the start of each epoch) and computes precision/recall/F1 per class from
    the FULL accumulated matrix only in `result()`. That is mathematically the
    same computation as `sklearn.metrics.f1_score(average="macro", zero_division=0)`
    on the concatenated full-epoch predictions - verified directly against
    `src.evaluation.compute_val_macro_f1` in `tests/test_models.py`.

    This is what makes `monitor="val_macro_f1"` in `ModelCheckpoint`/
    `EarlyStopping`/`ReduceLROnPlateau` correct: Keras computes it the same way
    for training-time monitoring as `src.evaluation` computes it for final
    reporting, because both reduce to the same confusion-matrix-based formula.
    """

    def __init__(self, num_classes: int, name: str = "macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion = self.add_weight(
            name="confusion", shape=(num_classes, num_classes), initializer="zeros", dtype="float32"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_pred_labels = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        batch_cm = tf.math.confusion_matrix(y_true, y_pred_labels, num_classes=self.num_classes, dtype=tf.float32)
        self.confusion.assign_add(batch_cm)

    def result(self):
        cm = self.confusion
        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp
        # zero_division=0 semantics, matching src.evaluation: a class with no
        # predicted and no true instances contributes F1=0 to the macro average,
        # not NaN.
        precision = tf.math.divide_no_nan(tp, tp + fp)
        recall = tf.math.divide_no_nan(tp, tp + fn)
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.confusion.assign(tf.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}


def build_recurrent_model(
    config: RecurrentModelConfig,
    embedding_matrix: Optional[np.ndarray] = None,
) -> keras.Model:
    """Input -> Embedding [-> SpatialDropout1D] -> [Bidirectional] LSTM -> Dense.

    Every M0-M4 difference is a config value, not a branch in this function
    beyond the two structural ones the ladder actually specifies: whether the
    LSTM is wrapped in `Bidirectional` (M1+) and whether `SpatialDropout1D` is
    present at all (M2+, since M0/M1 have `spatial_dropout=0.0`). Everything
    else - dropout rates, embedding source, `max_len`, learning rate - is a
    parameter, so M0 cannot silently end up stronger than the baseline it is
    supposed to be, and M3 cannot silently change anything besides the
    embedding.

    No masking (`mask_zero`) is used. `src.keras_tokenizer.pad_sequences`'
    `padding="pre"` default was chosen on the explicit assumption of an
    *unmasked* LSTM - see that module's docstring. Adding masking here would
    silently invalidate that reasoning, so it is deliberately not added.
    """
    if config.embedding_type == "glove":
        if embedding_matrix is None:
            raise ModelConfigError(f"{config.experiment_id} uses GloVe embeddings but no embedding_matrix was given")
        expected_shape = (config.vocab_size, config.embedding_dim)
        if embedding_matrix.shape != expected_shape:
            raise ModelConfigError(
                f"embedding_matrix shape {embedding_matrix.shape} does not match "
                f"(vocab_size, embedding_dim) = {expected_shape}"
            )
        embeddings_initializer = keras.initializers.Constant(embedding_matrix)
    else:
        if embedding_matrix is not None:
            raise ModelConfigError(
                f"{config.experiment_id} uses random embeddings; an embedding_matrix was "
                f"passed in anyway, which would silently change the experiment"
            )
        embeddings_initializer = "uniform"

    inputs = keras.Input(shape=(config.max_len,), dtype="int32", name="input_ids")
    x = keras.layers.Embedding(
        input_dim=config.vocab_size,
        output_dim=config.embedding_dim,
        embeddings_initializer=embeddings_initializer,
        trainable=config.embedding_trainable,
        name="embedding",
    )(inputs)

    if config.spatial_dropout > 0:
        x = keras.layers.SpatialDropout1D(config.spatial_dropout, name="spatial_dropout")(x)

    lstm = keras.layers.LSTM(
        config.lstm_units,
        dropout=config.dropout,
        recurrent_dropout=config.recurrent_dropout,
        name="lstm",
    )
    x = keras.layers.Bidirectional(lstm, name="bidirectional_lstm")(x) if config.bidirectional else lstm(x)

    outputs = keras.layers.Dense(config.output_classes, activation="softmax", name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{config.experiment_id}_recurrent")

    if config.optimizer != "adam":
        raise ModelConfigError(f"Unsupported optimizer '{config.optimizer}' - only 'adam' is used by M0-M4")
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    model.compile(
        optimizer=optimizer,
        # Integer class ids (src.data.CLASS_TO_ID / src.preprocessing.class_ids),
        # not one-hot - sparse_categorical_crossentropy is the loss that matches
        # that label representation.
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", MacroF1Score(num_classes=config.output_classes)],
    )
    return model


def build_callbacks(
    config: RecurrentModelConfig,
    checkpoint_path: Union[str, Path],
) -> list[keras.callbacks.Callback]:
    """Callback configuration only - nothing here calls `.fit()`.

    `ModelCheckpoint(monitor="val_macro_f1", save_best_only=True)` is held fixed
    across M0-M4 (project_plan.md §4's experimental controls) and is always
    included. `ReduceLROnPlateau` and `EarlyStopping` are added only when the
    config actually turns them on - M4 only, per the locked ladder.
    """
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    if config.lr_schedule == "reduce_on_plateau":
        if config.lr_schedule_factor is None or config.lr_schedule_patience is None or config.lr_schedule_min_lr is None:
            raise ModelConfigError(
                f"{config.experiment_id} sets lr_schedule='reduce_on_plateau' but is missing "
                f"lr_schedule_factor/lr_schedule_patience/lr_schedule_min_lr"
            )
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_macro_f1",
                mode="max",
                factor=config.lr_schedule_factor,
                patience=config.lr_schedule_patience,
                min_lr=config.lr_schedule_min_lr,
            )
        )

    if config.early_stopping:
        if config.early_stopping_patience is None:
            raise ModelConfigError(f"{config.experiment_id} sets early_stopping=True but early_stopping_patience is missing")
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_macro_f1",
                mode="max",
                patience=config.early_stopping_patience,
                restore_best_weights=config.early_stopping_restore_best_weights,
            )
        )

    return callbacks


def count_parameters(model: keras.Model) -> dict[str, int]:
    """Actual trainable/non-trainable weight counts from the built model - never
    hardcoded, so a config change is automatically reflected."""
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    non_trainable = int(sum(int(np.prod(w.shape)) for w in model.non_trainable_weights))
    return {"trainable": trainable, "non_trainable": non_trainable, "total": trainable + non_trainable}
