import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1,           bias=False)

    def forward(self, lstm_out):
        scores  = self.v(torch.tanh(self.W(lstm_out)))
        weights = F.softmax(scores, dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return context, weights.squeeze(-1)


class AttentionLSTM(nn.Module):

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int   = 128,
        hidden_dim:  int   = 256,
        num_layers:  int   = 2,
        dropout:     float = 0.5,
        pad_idx:     int   = 0,
        embedding:   nn.Embedding | None = None,
    ):
        super().__init__()

        if embedding is not None:
            self.embedding = embedding
            embed_dim      = embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim,
                                          padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_attn=False):
        embedded            = self.dropout(self.embedding(x))
        lstm_out, _         = self.lstm(embedded)
        lstm_out            = self.dropout(lstm_out)
        context, attn_w     = self.attention(lstm_out)
        logits              = self.fc(context).squeeze(1)
        if return_attn:
            return logits, attn_w
        return logits