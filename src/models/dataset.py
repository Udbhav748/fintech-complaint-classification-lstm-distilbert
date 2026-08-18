"""PyTorch Dataset and DataLoader preparation for LSTM models.

Encodes narrative text to integer token sequences using the training vocabulary,
applies deterministic padding/truncation to max_length, and yields PyTorch tensors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.vocabulary import Vocabulary, tokenize_text

logger = logging.getLogger(__name__)


class ComplaintDataset(Dataset):
    """PyTorch Dataset for FinTech Complaint sequences with pre-encoded tensors."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocabulary,
        max_length: int = 128,
    ) -> None:
        self.max_length = max_length
        num_samples = len(texts)

        # Pre-allocate tensor memory for fast batching
        input_ids_np = np.full((num_samples, max_length), vocab.pad_idx, dtype=np.int64)
        lengths_np = np.ones(num_samples, dtype=np.int64)

        for i, text in enumerate(texts):
            tokens = tokenize_text(text)
            token_ids = vocab.encode(tokens)
            act_len = min(len(token_ids), max_length)
            if act_len == 0:
                act_len = 1
                token_ids = [vocab.unk_idx]
            lengths_np[i] = act_len
            input_ids_np[i, :act_len] = token_ids[:act_len]

        self.input_ids = torch.from_numpy(input_ids_np)
        self.lengths = torch.from_numpy(lengths_np)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "length": self.lengths[idx],
            "label": self.labels[idx],
        }



def load_vocab_and_mappings(
    vocab_path: Path | str = "data/processed/lstm_vocab.json",
    label_mapping_path: Path | str = "data/processed/label_mapping.json",
) -> tuple[Vocabulary, dict[str, int], dict[int, str]]:
    """Load the Phase 4 LSTM vocabulary and label mappings."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    vocab = Vocabulary.from_dict(vocab_data["vocab"])

    with open(label_mapping_path, "r", encoding="utf-8") as f:
        id_to_name_raw = json.load(f)

    id_to_name = {int(k): str(v) for k, v in id_to_name_raw.items()}
    name_to_id = {v: k for k, v in id_to_name.items()}

    logger.info("Loaded vocabulary with %d words and label mapping with %d classes", vocab.size, len(id_to_name))
    return vocab, name_to_id, id_to_name


def create_dataloaders(
    train_path: Path | str,
    val_path: Path | str,
    test_path: Path | str,
    vocab: Vocabulary,
    name_to_id: dict[str, int],
    max_length: int = 128,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test splits."""
    logger.info("Loading processed splits from parquet/csv...")
    train_df = pd.read_parquet(train_path) if str(train_path).endswith(".parquet") else pd.read_csv(train_path)
    val_df = pd.read_parquet(val_path) if str(val_path).endswith(".parquet") else pd.read_csv(val_path)
    test_df = pd.read_parquet(test_path) if str(test_path).endswith(".parquet") else pd.read_csv(test_path)

    train_labels = [name_to_id[p] for p in train_df["product"]]
    val_labels = [name_to_id[p] for p in val_df["product"]]
    test_labels = [name_to_id[p] for p in test_df["product"]]

    train_dataset = ComplaintDataset(train_df["narrative"].tolist(), train_labels, vocab, max_length=max_length)
    val_dataset = ComplaintDataset(val_df["narrative"].tolist(), val_labels, vocab, max_length=max_length)
    test_dataset = ComplaintDataset(test_df["narrative"].tolist(), test_labels, vocab, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info(
        "Created DataLoaders: Train = %d samples (%d batches), Val = %d samples (%d batches), Test = %d samples (%d batches)",
        len(train_dataset),
        len(train_loader),
        len(val_dataset),
        len(val_loader),
        len(test_dataset),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader
