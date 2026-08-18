"""Simple LSTM Architecture for FinTech Complaint Classification.

Implements the E0 baseline:
  - Randomly initialized trainable embedding layer
  - Single-layer unidirectional LSTM
  - Dense linear classifier (5 classes)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SimpleLSTMClassifier(nn.Module):
    """Simple LSTM model for multiclass complaint classification.

    Architecture:
        Embedding(vocab_size, embedding_dim, padding_idx=0)
        LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        Linear(hidden_dim, num_classes)
    """

    def __init__(
        self,
        vocab_size: int = 25000,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 5,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        pad_idx: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification Head
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with token indices.
            lengths: Optional tensor of actual sequence lengths before padding.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)

        if lengths is not None:
            # Pack padded sequence for efficient computation and exact final state extraction
            lengths_cpu = lengths.to("cpu")
            # Ensure lengths are at least 1 to prevent packing errors
            lengths_clamped = torch.clamp(lengths_cpu, min=1)
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_clamped, batch_first=True, enforce_sorted=False
            )
            _, (hn, _) = self.lstm(packed_embedded)
        else:
            _, (hn, _) = self.lstm(embedded)

        # Extract final hidden state
        # hn shape: (num_layers * num_directions, batch_size, hidden_dim)
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = hn[-2, :, :]
            h_backward = hn[-1, :, :]
            h_final = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_final = hn[-1, :, :]

        # (batch_size, num_classes)
        logits = self.fc(h_final)
        return logits

    def count_parameters(self) -> dict[str, int]:
        """Return total, trainable, and non-trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total_params - trainable_params
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable,
        }
