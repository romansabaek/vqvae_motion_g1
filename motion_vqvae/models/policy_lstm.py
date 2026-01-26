"""
Simple LSTM model for policy ID prediction from motion sequences.
"""

import torch
import torch.nn as nn
from typing import Optional


class PolicyLSTM(nn.Module):
    """
    Simple LSTM model to predict policy IDs from motion sequences.
    
    Input: (batch_size, window_size, feature_dim) - normalized motion features
    Output: (batch_size, num_policies) - policy ID logits
    """
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_policies: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_policies = num_policies
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
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
            x: (batch_size, window_size, input_dim) - normalized motion features
        
        Returns:
            logits: (batch_size, num_policies) - policy ID logits
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, window_size, hidden_dim)
        
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

