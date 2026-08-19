"""Experiment metadata, environment tracking, and run logging utilities.

Provides explicit, transparent tracking for all model experiments (M0–M4, D0),
capturing environment details, git provenance, dataset/split fingerprints,
hyperparameters, parameter counts, training timings, and evaluation metrics.
"""

from dataclasses import asdict, dataclass, field
import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


def get_git_info(repo_dir: Union[str, Path] = ".") -> dict[str, Any]:
    """Extracts git commit hash, branch, and dirty status without raising errors."""
    info = {
        "git_commit": "unknown",
        "git_branch": "unknown",
        "git_dirty": False,
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_commit"] = commit
    except Exception:
        pass

    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_branch"] = branch
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_dirty"] = len(status) > 0
    except Exception:
        pass

    return info


def get_package_versions(packages: Optional[list[str]] = None) -> dict[str, str]:
    """Collects versions of core ML and data libraries."""
    if packages is None:
        packages = [
            "torch",
            "tensorflow",
            "transformers",
            "scikit-learn",
            "pandas",
            "numpy",
            "scipy",
            "accelerate",
            "tokenizers",
            "pyarrow",
        ]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
        except Exception as e:
            versions[pkg] = f"error: {e}"
    return versions


def get_device_info() -> dict[str, Any]:
    """Collects compute accelerator (GPU/CPU) information."""
    info: dict[str, Any] = {
        "device_type": "cpu",
        "device_name": platform.processor() or "Unknown CPU",
        "device_count": 1,
        "cuda_available": False,
        "cuda_version": None,
    }

    # Check PyTorch if available
    try:
        import torch

        if torch.cuda.is_available():
            info["device_type"] = "cuda"
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            return info
    except ImportError:
        pass

    # Check TensorFlow if available
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info["device_type"] = "gpu"
            info["device_name"] = gpus[0].name
            info["device_count"] = len(gpus)
            info["cuda_available"] = True
            return info
    except ImportError:
        pass

    return info


@dataclass
class EnvironmentMetadata:
    """Explicit record of execution platform and library environment."""
    python_version: str
    os_platform: str
    os_release: str
    git_commit: str
    git_branch: str
    git_dirty: bool
    package_versions: dict[str, str]
    device_info: dict[str, Any]

    @classmethod
    def capture(cls, repo_dir: Union[str, Path] = ".") -> "EnvironmentMetadata":
        git_info = get_git_info(repo_dir)
        return cls(
            python_version=sys.version.split(" ")[0],
            os_platform=platform.system(),
            os_release=platform.release(),
            git_commit=git_info["git_commit"],
            git_branch=git_info["git_branch"],
            git_dirty=git_info["git_dirty"],
            package_versions=get_package_versions(),
            device_info=get_device_info(),
        )


@dataclass
class EvaluationMetrics:
    """Structured evaluation metrics recorded on validation and test sets."""
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_macro_f1: Optional[float] = None
    val_macro_precision: Optional[float] = None
    val_macro_recall: Optional[float] = None
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_macro_f1: Optional[float] = None
    test_macro_precision: Optional[float] = None
    test_macro_recall: Optional[float] = None
    per_class_f1: Optional[dict[str, float]] = None
    per_class_precision: Optional[dict[str, float]] = None
    per_class_recall: Optional[dict[str, float]] = None


@dataclass
class ExperimentRecord:
    """Complete, explicit record of an experiment run."""
    run_id: str
    model_name: str  # e.g., "M0", "M1", "M2", "M3", "M4", "D0"
    seed: int
    started_at_utc: str
    ended_at_utc: str
    duration_seconds: float
    epochs_trained: int
    best_epoch: int
    trainable_params: int
    non_trainable_params: int
    total_params: int
    hyperparameters: dict[str, Any]
    dataset_content_sha256: str
    split_indices_sha256: dict[str, str]
    environment: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        return cls(**data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentRecord":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_runs_csv_row(self) -> dict[str, Any]:
        """Flattens the experiment record into the row format for results/runs.csv."""
        m = self.metrics
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "seed": self.seed,
            "train_loss": m.get("train_loss"),
            "val_loss": m.get("val_loss"),
            "val_accuracy": m.get("val_accuracy"),
            "best_val_macro_f1": m.get("val_macro_f1"),
            "val_macro_precision": m.get("val_macro_precision"),
            "val_macro_recall": m.get("val_macro_recall"),
            "test_loss": m.get("test_loss"),
            "test_accuracy": m.get("test_accuracy"),
            "test_macro_f1": m.get("test_macro_f1"),
            "test_macro_precision": m.get("test_macro_precision"),
            "test_macro_recall": m.get("test_macro_recall"),
            "best_epoch": self.best_epoch,
            "epochs_run": self.epochs_trained,
            "training_time_s": round(self.duration_seconds, 2),
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "dataset_content_sha256": self.dataset_content_sha256,
            "git_commit": self.environment.get("git_commit"),
            "timestamp_utc": self.ended_at_utc,
        }


class ExperimentTracker:
    """Manages experiment metadata creation, persistence, and summary generation."""

    def __init__(
        self,
        results_dir: Union[str, Path] = "results",
        runs_csv_name: str = "runs.csv",
        metadata_dirname: str = "metadata",
    ):
        self.results_dir = Path(results_dir)
        self.runs_csv_path = self.results_dir / runs_csv_name
        self.metadata_dir = self.results_dir / metadata_dirname
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        model_name: str,
        seed: int,
        started_at: datetime.datetime,
        ended_at: datetime.datetime,
        epochs_trained: int,
        best_epoch: int,
        trainable_params: int,
        total_params: int,
        hyperparameters: dict[str, Any],
        dataset_content_sha256: str,
        split_indices_sha256: dict[str, str],
        metrics: Union[EvaluationMetrics, dict[str, Any]],
        non_trainable_params: Optional[int] = None,
        artifacts: Optional[dict[str, str]] = None,
        notes: Optional[str] = None,
        run_id: Optional[str] = None,
        repo_dir: Union[str, Path] = ".",
    ) -> ExperimentRecord:
        """Creates, saves, and appends an ExperimentRecord."""
        if non_trainable_params is None:
            non_trainable_params = max(0, total_params - trainable_params)

        duration = (ended_at - started_at).total_seconds()
        timestamp_str = ended_at.strftime("%Y%m%d_%H%M%S")
        if run_id is None:
            run_id = f"{model_name}_seed{seed}_{timestamp_str}"

        env_meta = EnvironmentMetadata.capture(repo_dir=repo_dir)
        metrics_dict = asdict(metrics) if isinstance(metrics, EvaluationMetrics) else metrics

        record = ExperimentRecord(
            run_id=run_id,
            model_name=model_name,
            seed=seed,
            started_at_utc=started_at.isoformat(),
            ended_at_utc=ended_at.isoformat(),
            duration_seconds=duration,
            epochs_trained=epochs_trained,
            best_epoch=best_epoch,
            trainable_params=trainable_params,
            non_trainable_params=non_trainable_params,
            total_params=total_params,
            hyperparameters=hyperparameters,
            dataset_content_sha256=dataset_content_sha256,
            split_indices_sha256=split_indices_sha256,
            environment=asdict(env_meta),
            metrics=metrics_dict,
            artifacts=artifacts or {},
            notes=notes,
        )

        # 1. Save detailed JSON record
        json_path = self.metadata_dir / f"{run_id}.json"
        record.save(json_path)

        # 2. Append/update results/runs.csv
        self._append_to_runs_csv(record)

        return record

    def _append_to_runs_csv(self, record: ExperimentRecord) -> None:
        row = record.to_runs_csv_row()
        new_row_df = pd.DataFrame([row])

        if not self.runs_csv_path.exists():
            new_row_df.to_csv(self.runs_csv_path, index=False)
        else:
            existing_df = pd.read_csv(self.runs_csv_path)
            # If run_id exists, replace it, otherwise append
            if "run_id" in existing_df.columns and record.run_id in existing_df["run_id"].values:
                existing_df = existing_df[existing_df["run_id"] != record.run_id]
                updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            else:
                updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            updated_df.to_csv(self.runs_csv_path, index=False)

    def load_runs(self) -> pd.DataFrame:
        """Loads results/runs.csv as a pandas DataFrame."""
        if not self.runs_csv_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.runs_csv_path)

    def generate_comparison_table(
        self,
        model_order: Optional[list[str]] = None,
        model_descriptions: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Generates the comparison table mandated by Section 6 of project_plan.md.

        Computes:
        - Macro-F1 (reporting mean ± std for multi-seed models like M0 and D0)
        - Accuracy
        - Δ vs Previous
        - Δ vs M0
        """
        if model_order is None:
            model_order = ["M0", "M1", "M2", "M3", "M4", "D0"]

        if model_descriptions is None:
            model_descriptions = {
                "M0": "Unidirectional LSTM baseline, random embeddings",
                "M1": "Bidirectional LSTM",
                "M2": "BiLSTM + Spatial Dropout",
                "M3": "BiLSTM + Pretrained GloVe-100d",
                "M4": "BiLSTM + GloVe + LR schedule + EarlyStopping + max_len=256",
                "D0": "DistilBERT (fine-tuned transformer)",
            }

        df = self.load_runs()
        if df.empty:
            return pd.DataFrame(
                columns=["Model", "Configuration", "Macro-F1", "Accuracy", "Δ vs Previous", "Δ vs M0", "Interpretation"]
            )

        rows = []
        f1_by_model: dict[str, float] = {}

        for i, model in enumerate(model_order):
            sub = df[df["model_name"] == model]
            if sub.empty:
                continue

            desc = model_descriptions.get(model, "")
            num_runs = len(sub)

            # Metric values
            val_f1s = sub["best_val_macro_f1"].dropna()
            accs = sub["val_accuracy"].dropna()

            if val_f1s.empty:
                continue

            mean_f1 = float(val_f1s.mean())
            f1_by_model[model] = mean_f1
            mean_acc = float(accs.mean()) if not accs.empty else np.nan

            if num_runs > 1:
                std_f1 = float(val_f1s.std())
                f1_str = f"{mean_f1:.4f} ± {std_f1:.4f}"
                std_acc = float(accs.std()) if not accs.empty else 0.0
                acc_str = f"{mean_acc:.4f} ± {std_acc:.4f}"
            else:
                f1_str = f"{mean_f1:.4f}"
                acc_str = f"{mean_acc:.4f}" if not np.isnan(mean_acc) else "—"

            # Delta vs Previous
            if i == 0:
                delta_prev_str = "—"
            else:
                prev_model = model_order[i - 1]
                if prev_model in f1_by_model:
                    delta_prev = mean_f1 - f1_by_model[prev_model]
                    delta_prev_str = f"{'+' if delta_prev > 0 else ''}{delta_prev:.4f}"
                else:
                    delta_prev_str = "—"

            # Delta vs M0
            if i == 0 or "M0" not in f1_by_model:
                delta_m0_str = "—"
            else:
                delta_m0 = mean_f1 - f1_by_model["M0"]
                delta_m0_str = f"{'+' if delta_m0 > 0 else ''}{delta_m0:.4f}"

            rows.append({
                "Model": model,
                "Configuration": desc,
                "Macro-F1": f1_str,
                "Accuracy": acc_str,
                "Δ vs Previous": delta_prev_str,
                "Δ vs M0": delta_m0_str,
                "Interpretation": "",
            })

        return pd.DataFrame(rows)
