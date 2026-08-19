"""Lightweight structured logger for experiment training and evaluation."""

import datetime
import logging
from pathlib import Path
import sys
from typing import Any, Optional, Union


def get_logger(
    name: str = "cfpb_classifier",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Returns a configured logger with clean timestamped stdout and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            p = Path(log_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(p, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def log_experiment_start(
    logger: logging.Logger,
    experiment: str,
    seed: int,
    config: dict[str, Any],
) -> None:
    """Logs the kickoff of a model training run with key hyperparameters."""
    logger.info("=" * 60)
    logger.info(f"STARTING EXPERIMENT: {experiment} (Seed={seed})")
    logger.info("=" * 60)
    for k, v in sorted(config.items()):
        logger.info(f"  config.{k}: {v}")
    logger.info("-" * 60)


def log_epoch_summary(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    val_macro_f1: float,
    is_best: bool = False,
) -> None:
    """Logs a single epoch's losses and validation Macro-F1."""
    best_marker = " [★ BEST WEIGHTS SAVED]" if is_best else ""
    logger.info(
        f"Epoch {epoch:02d}/{total_epochs:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Macro-F1: {val_macro_f1:.4f}{best_marker}"
    )


def log_experiment_end(
    logger: logging.Logger,
    experiment: str,
    seed: int,
    best_epoch: int,
    best_val_macro_f1: float,
    train_time_s: float,
    checkpoint_path: Union[str, Path],
    test_macro_f1: Optional[float] = None,
) -> None:
    """Logs the final summary of an experiment run."""
    logger.info("-" * 60)
    logger.info(f"FINISHED EXPERIMENT: {experiment} (Seed={seed})")
    logger.info(f"  Best Epoch:           {best_epoch}")
    logger.info(f"  Best Val Macro-F1:    {best_val_macro_f1:.4f}")
    if test_macro_f1 is not None:
        logger.info(f"  Test Macro-F1:        {test_macro_f1:.4f}")
    logger.info(f"  Training Time:        {train_time_s:.2f}s ({train_time_s / 60:.2f} min)")
    logger.info(f"  Checkpoint Saved:     {checkpoint_path}")
    logger.info("=" * 60 + "\n")
