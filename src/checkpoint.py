"""Deterministic model checkpoint management and validation.

Enforces:
- Deterministic file paths per experiment and seed
- Saving and verification of checkpoint metadata
- Enforcement of `val_macro_f1` as the sole model selection metric
- Verification that best checkpoints exist and can be located for restoration
"""

from dataclasses import asdict, dataclass
import datetime
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

VALID_EXPERIMENTS = ["M0", "M1", "M2", "M3", "M4", "D0"]
MONITORED_METRIC = "val_macro_f1"


class CheckpointValidationError(ValueError):
    """Raised when checkpoint files or selection criteria violate project policy."""
    pass


@dataclass
class CheckpointMetadata:
    """Metadata recording the state and metric performance of a saved checkpoint."""
    experiment: str
    seed: int
    best_epoch: int
    monitor_metric: str
    best_metric_value: float
    checkpoint_file: str
    model_architecture: dict[str, Any]
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        return cls(**data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CheckpointMetadata":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint metadata file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_checkpoint_path(
    experiment: str,
    seed: int,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    extension: str = ".weights.h5",
) -> Path:
    """Returns deterministic file path for model weights."""
    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(
            f"Invalid experiment '{experiment}'. Must be one of {VALID_EXPERIMENTS}"
        )
    p = Path(checkpoint_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{experiment}_seed{seed}{extension}"


def get_metadata_path(
    experiment: str,
    seed: int,
    checkpoint_dir: Union[str, Path] = "checkpoints",
) -> Path:
    """Returns deterministic file path for checkpoint metadata JSON."""
    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(
            f"Invalid experiment '{experiment}'. Must be one of {VALID_EXPERIMENTS}"
        )
    p = Path(checkpoint_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{experiment}_seed{seed}_meta.json"


def save_checkpoint_record(
    experiment: str,
    seed: int,
    best_epoch: int,
    best_val_macro_f1: float,
    checkpoint_file: Union[str, Path],
    model_architecture: Optional[dict[str, Any]] = None,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    monitor_metric: str = MONITORED_METRIC,
) -> Path:
    """Records explicit checkpoint metadata to disk.

    Fails loudly if monitor_metric is not 'val_macro_f1'.
    """
    if monitor_metric != MONITORED_METRIC:
        raise CheckpointValidationError(
            f"Checkpoint monitoring metric must be '{MONITORED_METRIC}', got '{monitor_metric}'"
        )

    meta = CheckpointMetadata(
        experiment=experiment,
        seed=seed,
        best_epoch=best_epoch,
        monitor_metric=monitor_metric,
        best_metric_value=float(best_val_macro_f1),
        checkpoint_file=str(checkpoint_file),
        model_architecture=model_architecture or {},
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    meta_path = get_metadata_path(experiment, seed, checkpoint_dir=checkpoint_dir)
    meta.save(meta_path)
    return meta_path


def verify_checkpoint(
    experiment: str,
    seed: int,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    expected_extension: str = ".weights.h5",
    raise_on_failure: bool = True,
) -> tuple[bool, list[str]]:
    """Verifies that the checkpoint and its metadata exist and comply with project policy.

    Checks:
    1. Checkpoint file exists and is non-empty (>0 bytes).
    2. Checkpoint metadata exists.
    3. Monitored metric recorded in metadata is 'val_macro_f1'.
    4. Best epoch recorded is >= 1.
    5. Best metric value is within [0.0, 1.0].
    """
    errors: list[str] = []
    ckpt_path = get_checkpoint_path(experiment, seed, checkpoint_dir, extension=expected_extension)
    meta_path = get_metadata_path(experiment, seed, checkpoint_dir)

    if not ckpt_path.exists():
        errors.append(f"Checkpoint file not found: {ckpt_path}")
    elif ckpt_path.stat().st_size == 0:
        errors.append(f"Checkpoint file is 0 bytes (empty): {ckpt_path}")

    if not meta_path.exists():
        errors.append(f"Checkpoint metadata not found: {meta_path}")
    else:
        try:
            meta = CheckpointMetadata.load(meta_path)
            if meta.monitor_metric != MONITORED_METRIC:
                errors.append(
                    f"Invalid monitor metric in metadata: expected '{MONITORED_METRIC}', got '{meta.monitor_metric}'"
                )
            if meta.best_epoch < 1:
                errors.append(f"Invalid best epoch in metadata: {meta.best_epoch} (must be >= 1)")
            if not (0.0 <= meta.best_metric_value <= 1.0):
                errors.append(
                    f"Invalid best_val_macro_f1 in metadata: {meta.best_metric_value} (must be in [0, 1])"
                )
        except Exception as e:
            errors.append(f"Corrupt checkpoint metadata: {e}")

    is_valid = len(errors) == 0
    if not is_valid and raise_on_failure:
        raise CheckpointValidationError(
            f"Checkpoint verification failed for {experiment} (seed={seed}):\n"
            + "\n".join(f"- {e}" for e in errors)
        )
    return is_valid, errors
