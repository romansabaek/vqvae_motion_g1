"""
Sequence-wise Policy ID Classifier for VQ-VAE auxiliary loss.
Predicts policy IDs per timestep (sequence-to-sequence) instead of per window.

Input: codebook sequence [batch_size, seq_len]
Output: policy ID logits [batch_size, seq_len, num_policies] - one prediction per timestep
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SequencePolicyIDClassifier(nn.Module):
    """
    Sequence-to-sequence classifier that predicts policy IDs for each timestep.
    
    Input: codebook indices [batch_size, seq_len]
    Output: policy ID logits [batch_size, seq_len, num_policies] - one per timestep
    """
    
    def __init__(
        self,
        num_codebooks: int,  # Vocabulary size of codebook
        num_policies: int,  # Number of distinct policy IDs
        code_dim: int = 512,  # Embedding dimension for codebook indices
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        architecture: str = "lstm",  # "lstm", "transformer", or "cnn1d"
        num_heads: int = 8,  # For transformer: number of attention heads
        kernel_size: int = 3,  # For CNN1D: kernel size for convolutions
    ):
        super().__init__()
        
        self.num_codebooks = num_codebooks
        self.num_policies = num_policies
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture.lower()
        self.kernel_size = kernel_size
        
        if self.architecture not in ["lstm", "transformer", "cnn1d"]:
            raise ValueError(f"architecture must be 'lstm', 'transformer', or 'cnn1d', got '{architecture}'")
        
        # Embed codebook indices
        self.codebook_embedding = nn.Embedding(num_codebooks, code_dim)
        
        if self.architecture == "lstm":
            # LSTM: Sequential processing, outputs per timestep
            self.lstm = nn.LSTM(
                input_size=code_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
            # Per-timestep classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_policies)
            )
            self.transformer = None
            self.cnn1d = None
            
        elif self.architecture == "transformer":
            # Transformer: Self-attention, outputs per timestep
            self.pos_encoding = PositionalEncoding(code_dim, dropout)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=code_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Per-timestep classifier
            self.classifier = nn.Sequential(
                nn.Linear(code_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_policies)
            )
            self.lstm = None
            self.cnn1d = None
            
        elif self.architecture == "cnn1d":
            # 1D CNN: Convolutional layers with causal padding for sequence prediction
            cnn_layers = []
            in_channels = code_dim
            
            for i in range(num_layers):
                cnn_layers.append(nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size - 1,  # Causal padding (pad on left)
                    stride=1
                ))
                cnn_layers.append(nn.BatchNorm1d(hidden_dim))
                cnn_layers.append(nn.ReLU())
                cnn_layers.append(nn.Dropout(dropout))
                in_channels = hidden_dim
            
            self.cnn1d = nn.Sequential(*cnn_layers)
            
            # Per-timestep classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_policies)
            )
            self.lstm = None
            self.transformer = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.codebook_embedding.weight)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, codebook_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            codebook_seq: Codebook indices [batch_size, seq_len]
        
        Returns:
            policy_logits: Policy ID logits [batch_size, seq_len, num_policies]
        """
        batch_size, seq_len = codebook_seq.shape
        
        # Embed codebook indices: [batch_size, seq_len] -> [batch_size, seq_len, code_dim]
        embedded = self.codebook_embedding(codebook_seq)
        
        if self.architecture == "lstm":
            # LSTM: Sequential processing, outputs per timestep
            # [batch_size, seq_len, code_dim] -> [batch_size, seq_len, hidden_dim]
            lstm_out, _ = self.lstm(embedded)
            # Apply classifier to each timestep
            # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_policies]
            policy_logits = self.classifier(lstm_out)
            
        elif self.architecture == "transformer":
            # Transformer: Self-attention, outputs per timestep
            # Add positional encoding
            embedded = self.pos_encoding(embedded)
            # [batch_size, seq_len, code_dim] -> [batch_size, seq_len, code_dim]
            transformer_out = self.transformer(embedded)
            # Apply classifier to each timestep
            # [batch_size, seq_len, code_dim] -> [batch_size, seq_len, num_policies]
            policy_logits = self.classifier(transformer_out)
            
        elif self.architecture == "cnn1d":
            # 1D CNN: Convolutional layers with causal padding
            # Permute for Conv1d: [batch_size, seq_len, code_dim] -> [batch_size, code_dim, seq_len]
            embedded_permuted = embedded.permute(0, 2, 1)
            # Apply CNN: [batch_size, code_dim, seq_len] -> [batch_size, hidden_dim, seq_len + padding]
            cnn_out = self.cnn1d(embedded_permuted)
            # Remove extra padding from causal convolution
            cnn_out = cnn_out[:, :, :seq_len]  # [batch_size, hidden_dim, seq_len]
            # Permute back: [batch_size, hidden_dim, seq_len] -> [batch_size, seq_len, hidden_dim]
            cnn_out = cnn_out.permute(0, 2, 1)
            # Apply classifier to each timestep
            # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_policies]
            policy_logits = self.classifier(cnn_out)
        
        return policy_logits
    
    def predict(self, codebook_seq: torch.Tensor) -> torch.Tensor:
        """
        Predict policy IDs from codebook sequences.
        
        Args:
            codebook_seq: Codebook indices [batch_size, seq_len]
        
        Returns:
            policy_ids: Predicted policy IDs [batch_size, seq_len] - one per timestep
        """
        logits = self.forward(codebook_seq)
        return torch.argmax(logits, dim=2)  # [batch_size, seq_len]


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    Adds positional information to embeddings.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

