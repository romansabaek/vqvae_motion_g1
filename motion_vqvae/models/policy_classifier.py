"""
Policy ID Classifier for VQ-VAE auxiliary loss.
Takes codebook sequences as input and predicts policy IDs.
Supports four architectures: MLP, 1D CNN, LSTM, and Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PolicyIDClassifier(nn.Module):
    """
    Classifier that predicts policy IDs from codebook sequences.
    Supports four architectures:
    - MLP: Simple average pooling + MLP (fastest, no temporal modeling)
    - CNN1D: 1D Convolutional layers (fast, captures local temporal patterns)
    - LSTM: Sequential processing with LSTM (good temporal modeling, moderate speed)
    - Transformer: Self-attention mechanism (best for long-range dependencies, slower)
    
    Input: codebook indices [batch_size, seq_len]
    Output: policy ID logits [batch_size, num_policies]
    """
    
    def __init__(
        self,
        num_codebooks: int,  # Vocabulary size of codebook
        num_policies: int,  # Number of distinct policy IDs
        code_dim: int = 512,  # Embedding dimension for codebook indices
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        architecture: str = "lstm",  # "mlp", "cnn1d", "lstm", or "transformer"
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
        
        if self.architecture not in ["mlp", "cnn1d", "lstm", "transformer"]:
            raise ValueError(f"architecture must be 'mlp', 'cnn1d', 'lstm', or 'transformer', got '{architecture}'")
        
        # Embed codebook indices
        self.codebook_embedding = nn.Embedding(num_codebooks, code_dim)
        
        if self.architecture == "mlp":
            # Simple MLP classifier with average pooling (no temporal modeling)
            # Fastest, but loses temporal information
            layers = []
            input_dim = code_dim
            for i in range(num_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            
            # Final classification layer
            layers.append(nn.Linear(input_dim, num_policies))
            self.classifier = nn.Sequential(*layers)
            self.cnn1d = None
            self.lstm = None
            self.transformer = None
            
        elif self.architecture == "cnn1d":
            # 1D CNN: Convolutional layers for local temporal pattern extraction
            # Fast, parallelizable, captures local dependencies
            # Input: [batch_size, seq_len, code_dim] -> need to permute for Conv1d
            # Conv1d expects: [batch_size, channels, seq_len]
            cnn_layers = []
            in_channels = code_dim
            
            for i in range(num_layers):
                cnn_layers.append(nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # Same padding
                    stride=1
                ))
                cnn_layers.append(nn.BatchNorm1d(hidden_dim))
                cnn_layers.append(nn.ReLU())
                cnn_layers.append(nn.Dropout(dropout))
                in_channels = hidden_dim
            
            # Global average pooling over sequence dimension
            cnn_layers.append(nn.AdaptiveAvgPool1d(1))  # [batch_size, hidden_dim, 1]
            
            self.cnn1d = nn.Sequential(*cnn_layers)
            
            # MLP classifier on CNN output
            layers = []
            input_dim = hidden_dim
            for i in range(num_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            
            # Final classification layer
            layers.append(nn.Linear(input_dim, num_policies))
            self.classifier = nn.Sequential(*layers)
            self.lstm = None
            self.transformer = None
            
        elif self.architecture == "lstm":
            # LSTM: Sequential processing, good for temporal patterns
            # Moderate speed, captures sequential dependencies
            self.lstm = nn.LSTM(
                input_size=code_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
            # MLP classifier on LSTM output
            layers = []
            input_dim = hidden_dim
            for i in range(num_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            
            # Final classification layer
            layers.append(nn.Linear(input_dim, num_policies))
            self.classifier = nn.Sequential(*layers)
            self.cnn1d = None
            self.transformer = None
            
        elif self.architecture == "transformer":
            # Transformer: Self-attention mechanism
            # Best for long-range dependencies, parallelizable, but slower
            # BERT-style CLS token: learnable embedding prepended to sequence
            self.cls_token = nn.Parameter(torch.randn(1, 1, code_dim))  # [1, 1, code_dim] - learnable CLS token
            
            # Positional encoding (accounts for CLS token + sequence)
            self.pos_encoding = PositionalEncoding(code_dim, dropout, max_len=5001)  # +1 for CLS token
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=code_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # MLP classifier on CLS token output
            layers = []
            input_dim = code_dim
            for i in range(num_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            
            # Final classification layer
            layers.append(nn.Linear(input_dim, num_policies))
            self.classifier = nn.Sequential(*layers)
            self.cnn1d = None
            self.lstm = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.codebook_embedding.weight)
        # Initialize CLS token (if exists)
        if hasattr(self, 'cls_token'):
            nn.init.normal_(self.cls_token, std=0.02)  # BERT-style initialization
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
            policy_logits: Policy ID logits [batch_size, num_policies]
        """
        batch_size, seq_len = codebook_seq.shape
        
        # Embed codebook indices: [batch_size, seq_len] -> [batch_size, seq_len, code_dim]
        embedded = self.codebook_embedding(codebook_seq)
        
        if self.architecture == "mlp":
            # Average pooling: loses temporal information but fast
            # [batch_size, seq_len, code_dim] -> [batch_size, code_dim]
            pooled = embedded.mean(dim=1)
            feature_dim = self.code_dim
            
        elif self.architecture == "cnn1d":
            # 1D CNN: Convolutional layers for local temporal patterns
            # Permute for Conv1d: [batch_size, seq_len, code_dim] -> [batch_size, code_dim, seq_len]
            embedded_permuted = embedded.permute(0, 2, 1)
            # Apply CNN: [batch_size, code_dim, seq_len] -> [batch_size, hidden_dim, 1]
            cnn_out = self.cnn1d(embedded_permuted)
            # Squeeze: [batch_size, hidden_dim, 1] -> [batch_size, hidden_dim]
            pooled = cnn_out.squeeze(-1)
            feature_dim = self.hidden_dim
            
        elif self.architecture == "lstm":
            # LSTM: Sequential processing, captures temporal patterns
            # [batch_size, seq_len, code_dim] -> [batch_size, seq_len, hidden_dim]
            lstm_out, (h_n, c_n) = self.lstm(embedded)
            # Use the last hidden state: [batch_size, hidden_dim]
            pooled = lstm_out[:, -1, :]  # Take last timestep output
            feature_dim = self.hidden_dim
            
        elif self.architecture == "transformer":
            # Transformer: Self-attention, best for long-range dependencies
            # BERT-style: Prepend learnable CLS token to sequence
            # Expand CLS token to batch size: [1, 1, code_dim] -> [batch_size, 1, code_dim]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            # Concatenate CLS token with embedded sequence: [batch_size, 1+seq_len, code_dim]
            embedded_with_cls = torch.cat([cls_tokens, embedded], dim=1)
            
            # Add positional encoding (includes CLS token position)
            embedded_with_cls = self.pos_encoding(embedded_with_cls)
            
            # Pass through transformer: [batch_size, 1+seq_len, code_dim] -> [batch_size, 1+seq_len, code_dim]
            transformer_out = self.transformer(embedded_with_cls)
            
            # Extract CLS token (first token): [batch_size, code_dim]
            pooled = transformer_out[:, 0, :]  # CLS token aggregates sequence information
            feature_dim = self.code_dim
        
        # Classify: [batch_size, feature_dim] -> [batch_size, num_policies]
        logits = self.classifier(pooled)
        
        return logits
    
    def predict(self, codebook_seq: torch.Tensor) -> torch.Tensor:
        """
        Predict policy IDs from codebook sequences.
        
        Args:
            codebook_seq: Codebook indices [batch_size, seq_len]
        
        Returns:
            policy_ids: Predicted policy IDs [batch_size]
        """
        logits = self.forward(codebook_seq)
        return torch.argmax(logits, dim=1)


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

