"""
LSTM model for policy ID prediction from VQVAE codebook sequences.
Uses embedding layer for codebook indices.
"""

import torch
import torch.nn as nn
from typing import Optional


class PolicyLSTMCodebook(nn.Module):
    """
    LSTM model to predict policy IDs from codebook sequences.
    
    Input: (batch_size, window_size) - codebook indices (integers)
    Output: (batch_size, num_policies) - policy ID logits
    """
    
    def __init__(
        self,
        codebook_size: int = 512,  # Number of codebooks
        embedding_dim: int = 64,   # Embedding dimension for codebook indices
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_policies: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_policies = num_policies
        self.bidirectional = bidirectional
        
        # Embedding layer for codebook indices
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_policies)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, window_size) - codebook indices (long tensor)
        
        Returns:
            logits: (batch_size, num_policies) - policy ID logits
        """
        # Embed codebook indices
        x_emb = self.embedding(x)  # (batch_size, window_size, embedding_dim)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x_emb)  # lstm_out: (batch_size, window_size, hidden_dim)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            h_forward = h_n[-2]  # Last forward layer
            h_backward = h_n[-1]  # Last backward layer
            h_final = torch.cat([h_forward, h_backward], dim=1)  # (batch_size, hidden_dim * 2)
        else:
            h_final = h_n[-1]  # (batch_size, hidden_dim)
        
        # Project to policy logits
        logits = self.fc(h_final)  # (batch_size, num_policies)
        
        return logits

