import torch.nn as nn


class VanillaRNN(nn.Module):

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int   = 128,
        hidden_dim:  int   = 256,
        num_layers:  int   = 2,
        dropout:     float = 0.5,
        pad_idx:     int   = 0,
        embedding:   nn.Embedding | None = None,   # ← injected externally
    ):
        super().__init__()

        # Use pre-built embedding or create a fresh one
        if embedding is not None:
            self.embedding = embedding
            embed_dim      = embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim,
                                          padding_idx=pad_idx)

        self.rnn = nn.RNN(
            input_size   = embed_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            nonlinearity = "tanh",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded    = self.dropout(self.embedding(x))
        _, hidden   = self.rnn(embedded)
        out         = self.dropout(hidden[-1])
        return self.fc(out).squeeze(1)