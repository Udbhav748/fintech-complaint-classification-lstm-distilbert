"""Reproducibility utilities for seed control and software/environment capture.

Ensures deterministic execution across Python, NumPy, PyTorch, and TensorFlow,
and records explicit software environment details.
"""

from dataclasses import asdict, dataclass
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Optional, Union

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets random seeds deterministically across all random sources used in the project.

    Covers:
    - Python built-in `random`
    - Python hash seed (`os.environ["PYTHONHASHSEED"]`)
    - NumPy (`np.random.seed`)
    - PyTorch (CPU and CUDA if installed)
    - TensorFlow (if installed)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
    except ImportError:
        pass


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
    """Collects installed package versions."""
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
    """Collects CPU and GPU accelerator hardware info."""
    info: dict[str, Any] = {
        "device_type": "cpu",
        "device_name": platform.processor() or "Unknown CPU",
        "device_count": 1,
        "cuda_available": False,
        "cuda_version": None,
    }
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
class EnvironmentInfo:
    """Explicit, lightweight container for environment provenance."""
    python_version: str
    os_platform: str
    git_commit: str
    git_branch: str
    git_dirty: bool
    packages: dict[str, str]
    device: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def capture(cls, repo_dir: Union[str, Path] = ".") -> "EnvironmentInfo":
        git = get_git_info(repo_dir)
        return cls(
            python_version=sys.version.split(" ")[0],
            os_platform=f"{platform.system()} {platform.release()}",
            git_commit=git["git_commit"],
            git_branch=git["git_branch"],
            git_dirty=git["git_dirty"],
            packages=get_package_versions(),
            device=get_device_info(),
        )
