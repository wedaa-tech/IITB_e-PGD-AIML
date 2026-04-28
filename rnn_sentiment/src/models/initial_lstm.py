"""
LSTM for binary sentiment classification.

Architecture:
  Embedding → Dropout → LSTM (1 or 2 layers) → last hidden state → FC → Sigmoid

Key difference from VanillaRNN: LSTM adds a cell state (c_t) alongside
the hidden state (h_t), controlled by input / forget / output gates.
This mitigates the vanishing gradient problem for longer sequences.
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int   = 128,
        hidden_dim:   int   = 256,
        num_layers:   int   = 2,
        dropout:      float = 0.5,
        pad_idx:      int   = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ── Embedding ─────────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        # ── Recurrent core ────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ── Classifier head ───────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded        = self.dropout(self.embedding(x))  # (B, S, E)
        _, (hidden, _)  = self.lstm(embedded)              # hidden: (layers, B, H)

        out = self.dropout(hidden[-1])                     # (B, H)
        return self.fc(out).squeeze(1)                     # (B,)