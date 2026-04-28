"""
Attention-augmented LSTM for binary sentiment classification.

Architecture:
  Embedding → Dropout
    → LSTM (all timestep outputs retained)
    → Additive Attention over all outputs
    → Context vector (weighted sum)
    → FC → Sigmoid

Why attention helps:
  Instead of relying only on the last hidden state, the attention mechanism
  learns a soft importance weight for EVERY token's output. The final
  representation is a weighted sum (context vector), allowing the model to
  focus on the most sentiment-relevant parts of the review regardless of
  where they appear in the sequence.

Attention mechanism (Bahdanau-style, single-layer):
  score(h_t) = v · tanh(W · h_t)          -- alignment score
  α_t        = softmax(score(h_t))         -- attention weight
  context    = Σ α_t · h_t                 -- context vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Single-head additive (Bahdanau) attention over all LSTM output states.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1,           bias=False)

    def forward(self, lstm_out):
        # lstm_out : (B, S, H)
        scores  = self.v(torch.tanh(self.W(lstm_out)))  # (B, S, 1)
        weights = F.softmax(scores, dim=1)              # (B, S, 1)
        context = (weights * lstm_out).sum(dim=1)       # (B, H)
        return context, weights.squeeze(-1)             # context, attn map


class AttentionLSTM(nn.Module):

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

        # ── Attention ─────────────────────────────────────────────────────
        self.attention = Attention(hidden_dim)

        # ── Classifier head ───────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_attn=False):
        embedded  = self.dropout(self.embedding(x))    # (B, S, E)
        lstm_out, _ = self.lstm(embedded)              # (B, S, H)
        lstm_out  = self.dropout(lstm_out)

        context, attn_weights = self.attention(lstm_out)  # (B,H), (B,S)
        logits = self.fc(context).squeeze(1)              # (B,)

        if return_attn:
            return logits, attn_weights
        return logits