"""
Sequence-wise Policy ID Classifier for VQ-VAE auxiliary loss.

Goal:
- Input:  codebook indices [batch_size, seq_len]
- Output: per-timestep policy logits [batch_size, seq_len, num_policies]

This is used when config['policy_use_sequence_wise'] == True in MVQVAEAgent.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class SequencePolicyIDClassifier(nn.Module):
    """
    Sequence-wise classifier that predicts policy ID logits for each timestep.

    Architectures:
      - "mlp": token-wise MLP (no temporal modeling; fast baseline)
      - "cnn1d": temporal conv over embeddings, then token-wise head
      - "lstm": LSTM over embeddings, then token-wise head
      - "transformer": Transformer encoder, then token-wise head

    Input:  codebook_seq [B, T]
    Output: policy_logits [B, T, C]
    """

    def __init__(
        self,
        num_codebooks: int,
        num_policies: int,
        code_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        architecture: str = "lstm",
        num_heads: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_policies = num_policies
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.architecture = architecture.lower()
        self.num_heads = num_heads
        self.kernel_size = kernel_size

        if self.architecture not in ["mlp", "cnn1d", "lstm", "transformer"]:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.codebook_embedding = nn.Embedding(num_codebooks, code_dim)

        if self.architecture == "mlp":
            # Token-wise MLP: independently classify each timestep
            layers = []
            in_dim = code_dim
            for _ in range(max(1, num_layers)):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, num_policies)

        elif self.architecture == "cnn1d":
            # Conv1d over time: [B, D, T] -> [B, H, T]
            cnn = []
            in_ch = code_dim
            for _ in range(max(1, num_layers)):
                cnn.append(nn.Conv1d(in_ch, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2))
                cnn.append(nn.BatchNorm1d(hidden_dim))
                cnn.append(nn.ReLU())
                cnn.append(nn.Dropout(dropout))
                in_ch = hidden_dim
            self.backbone = nn.Sequential(*cnn)
            self.head = nn.Linear(hidden_dim, num_policies)

        elif self.architecture == "lstm":
            self.lstm = nn.LSTM(
                input_size=code_dim,
                hidden_size=hidden_dim,
                num_layers=max(1, num_layers),
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=False,
            )
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(hidden_dim, num_policies)

        elif self.architecture == "transformer":
            self.pos_encoding = PositionalEncoding(code_dim, dropout=dropout, max_len=5001)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=code_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=max(1, num_layers))
            self.proj = nn.Sequential(
                nn.Linear(code_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(hidden_dim, num_policies)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.codebook_embedding.weight)
        if hasattr(self, "head") and isinstance(self.head, nn.Linear):
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, codebook_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codebook_seq: [B, T] (Long)

        Returns:
            policy_logits: [B, T, C]
        """
        if codebook_seq.dtype != torch.long:
            codebook_seq = codebook_seq.long()

        B, T = codebook_seq.shape
        x = self.codebook_embedding(codebook_seq)  # [B, T, D]

        if self.architecture == "mlp":
            # Token-wise MLP
            h = self.backbone(x)  # [B, T, H]
            logits = self.head(h)  # [B, T, C]
            return logits

        if self.architecture == "cnn1d":
            # [B, T, D] -> [B, D, T]
            x_c = x.permute(0, 2, 1)
            h_c = self.backbone(x_c)  # [B, H, T]
            h = h_c.permute(0, 2, 1)  # [B, T, H]
            logits = self.head(h)     # [B, T, C]
            return logits

        if self.architecture == "lstm":
            lstm_out, _ = self.lstm(x)   # [B, T, H]
            h = self.proj(lstm_out)      # [B, T, H]
            logits = self.head(h)        # [B, T, C]
            return logits

        # transformer
        x_pe = self.pos_encoding(x)         # [B, T, D]
        t_out = self.transformer(x_pe)      # [B, T, D]
        h = self.proj(t_out)                # [B, T, H]
        logits = self.head(h)               # [B, T, C]
        return logits
