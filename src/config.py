"""Configuration loading and validation utilities.

Provides explicit, transparent loading and validation for dataset, tokenizer,
and experiment configurations without hidden defaults or global mutations.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

from src.checkpoint import VALID_EXPERIMENTS
from src.data import LABELS

DATA_CONFIG_PATH = Path("configs/data_config.json")

# Pre-registered default configurations for M0–M4, D0 matching project_plan.md
EXPERIMENT_CONFIGS: dict[str, dict[str, Any]] = {
    "M0": {
        "experiment": "M0",
        "model_type": "lstm",
        "description": "Unidirectional LSTM baseline, random embeddings",
        "embedding_dim": 100,
        "hidden_dim": 128,
        "num_layers": 1,
        "bidirectional": False,
        "dropout": 0.0,
        "spatial_dropout": 0.0,
        "use_glove": False,
        "glove_trainable": True,
        "max_len": 128,
        "vocab_size": 20000,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "epochs": 10,
        "early_stopping": False,
        "seeds": [42, 123, 456],
    },
    "M1": {
        "experiment": "M1",
        "model_type": "bilstm",
        "description": "Bidirectional LSTM",
        "embedding_dim": 100,
        "hidden_dim": 128,
        "num_layers": 1,
        "bidirectional": True,
        "dropout": 0.0,
        "spatial_dropout": 0.0,
        "use_glove": False,
        "glove_trainable": True,
        "max_len": 128,
        "vocab_size": 20000,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "epochs": 10,
        "early_stopping": False,
        "seeds": [42],
    },
    "M2": {
        "experiment": "M2",
        "model_type": "bilstm_dropout",
        "description": "BiLSTM + Spatial Dropout",
        "embedding_dim": 100,
        "hidden_dim": 128,
        "num_layers": 1,
        "bidirectional": True,
        "dropout": 0.3,
        "spatial_dropout": 0.2,
        "use_glove": False,
        "glove_trainable": True,
        "max_len": 128,
        "vocab_size": 20000,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "epochs": 10,
        "early_stopping": False,
        "seeds": [42],
    },
    "M3": {
        "experiment": "M3",
        "model_type": "bilstm_glove",
        "description": "BiLSTM + Pretrained GloVe-100d",
        "embedding_dim": 100,
        "hidden_dim": 128,
        "num_layers": 1,
        "bidirectional": True,
        "dropout": 0.3,
        "spatial_dropout": 0.2,
        "use_glove": True,
        "glove_path": "data/embeddings/glove.6B.100d.txt",
        "glove_trainable": True,
        "max_len": 128,
        "vocab_size": 20000,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "epochs": 10,
        "early_stopping": False,
        "seeds": [42],
    },
    "M4": {
        "experiment": "M4",
        "model_type": "bilstm_optimized",
        "description": "BiLSTM + GloVe + LR schedule + EarlyStopping + max_len=256",
        "embedding_dim": 100,
        "hidden_dim": 128,
        "num_layers": 1,
        "bidirectional": True,
        "dropout": 0.3,
        "spatial_dropout": 0.2,
        "use_glove": True,
        "glove_path": "data/embeddings/glove.6B.100d.txt",
        "glove_trainable": True,
        "max_len": 256,
        "vocab_size": 20000,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "lr_schedule": "reduce_on_plateau",
        "optimizer": "adam",
        "epochs": 20,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "seeds": [42],
    },
    "D0": {
        "experiment": "D0",
        "model_type": "distilbert",
        "description": "DistilBERT (fine-tuned transformer)",
        "pretrained_model_name": "distilbert-base-uncased",
        "max_len": 256,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "epochs": 4,
        "early_stopping": True,
        "early_stopping_patience": 2,
        "seeds": [42, 123, 456],
    },
}


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


def load_data_config(config_path: Union[str, Path] = DATA_CONFIG_PATH) -> dict[str, Any]:
    """Loads and validates the central data configuration JSON."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Data config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Basic schema validation
    required_sections = ["dataset", "cleaning", "split"]
    for sec in required_sections:
        if sec not in cfg:
            raise ConfigValidationError(f"Missing required section '{sec}' in {config_path}")

    return cfg


def get_experiment_config(experiment: str) -> dict[str, Any]:
    """Returns a deep-copy dictionary of pre-registered experiment configuration."""
    if experiment not in VALID_EXPERIMENTS:
        raise ConfigValidationError(
            f"Invalid experiment identifier '{experiment}'. Must be one of {VALID_EXPERIMENTS}."
        )
    return json.loads(json.dumps(EXPERIMENT_CONFIGS[experiment]))


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validates an experiment configuration dictionary."""
    if "experiment" not in config:
        raise ConfigValidationError("Configuration missing required 'experiment' key.")

    exp = config["experiment"]
    if exp not in VALID_EXPERIMENTS:
        raise ConfigValidationError(
            f"Invalid experiment '{exp}'. Must strictly be one of {VALID_EXPERIMENTS}."
        )

    required_keys = ["model_type", "batch_size", "learning_rate", "epochs"]
    for k in required_keys:
        if k not in config:
            raise ConfigValidationError(f"Missing required parameter '{k}' in {exp} config.")

    if config["learning_rate"] <= 0:
        raise ConfigValidationError(f"Learning rate must be positive, got {config['learning_rate']}")

    if config["epochs"] < 1:
        raise ConfigValidationError(f"Epochs must be >= 1, got {config['epochs']}")
