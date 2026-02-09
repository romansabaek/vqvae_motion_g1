"""
Motion Predictor for VQ-VAE auxiliary loss.
Predicts future raw motion features (e.g., 0.5 seconds ahead) from codebook sequences.
Supports multiple architectures: MLP (simple), LSTM, Transformer, and CNN1D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MotionPredictor(nn.Module):
    """
    Predictor that forecasts future raw motion features from current codebook sequences.
    Supports multiple architectures for temporal modeling:
    - MLP: Simple average pooling + MLP (baseline, loses temporal info)
    - LSTM: Sequential processing (good temporal modeling)
    - Transformer: Self-attention (best for long-range dependencies)
    - CNN1D: Convolutional layers (fast, captures local patterns)
    
    Input: codebook indices [batch_size, seq_len]
    Output: predicted raw motion features [batch_size, pred_frames, frame_size]
    """
    
    def __init__(
        self,
        num_codebooks: int,  # Vocabulary size of codebook
        code_dim: int = 512,  # Embedding dimension for codebook indices
        frame_size: int = 50,  # Size of motion feature vector (e.g., 50 for G1 humanoid)
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pred_frames: int = 15,  # Number of future frames to predict (e.g., 15 for 0.5s at 30 FPS)
        architecture: str = "lstm",  # "mlp", "lstm", "transformer", or "cnn1d"
        num_heads: int = 8,  # For transformer: number of attention heads
        kernel_size: int = 3,  # For CNN1D: kernel size for convolutions
    ):
        super().__init__()
        
        self.num_codebooks = num_codebooks
        self.code_dim = code_dim
        self.frame_size = frame_size
        self.hidden_dim = hidden_dim
        self.pred_frames = pred_frames
        self.architecture = architecture.lower()
        
        if self.architecture not in ["mlp", "lstm", "transformer", "cnn1d"]:
            raise ValueError(f"architecture must be 'mlp', 'lstm', 'transformer', or 'cnn1d', got '{architecture}'")
        
        # Embed codebook indices
        self.codebook_embedding = nn.Embedding(num_codebooks, code_dim)
        
        if self.architecture == "mlp":
            # Simple MLP: average pool sequence then predict (baseline)
            layers = []
            input_dim = code_dim
            for i in range(num_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            
            # Output layer: predict future motion features
            layers.append(nn.Linear(input_dim, frame_size * pred_frames))
            self.predictor = nn.Sequential(*layers)
            self.lstm = None
            self.transformer = None
            self.cnn1d = None
            self.pos_encoding = None
            
        elif self.architecture == "lstm":
            # LSTM: Sequential processing, captures temporal patterns
            self.lstm = nn.LSTM(
                input_size=code_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
            # Predictor head: takes last hidden state
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, frame_size * pred_frames)
            )
            self.transformer = None
            self.cnn1d = None
            self.pos_encoding = None
            
        elif self.architecture == "transformer":
            # Transformer: Self-attention, best for long-range dependencies
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
            
            # Predictor head: uses last token or average pooling
            self.predictor = nn.Sequential(
                nn.Linear(code_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, frame_size * pred_frames)
            )
            self.lstm = None
            self.cnn1d = None
            
        elif self.architecture == "cnn1d":
            # 1D CNN: Convolutional layers for local temporal patterns
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
            
            # Predictor head
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, frame_size * pred_frames)
            )
            self.lstm = None
            self.transformer = None
            self.pos_encoding = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.codebook_embedding.weight)
        for module in self.predictor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, codebook_seq: torch.Tensor) -> torch.Tensor:
        """
        Predict future raw motion features.
        
        Args:
            codebook_seq: Current codebook indices [batch_size, seq_len]
        
        Returns:
            pred_motion: Predicted raw motion features [batch_size, pred_frames, frame_size]
        """
        batch_size, seq_len = codebook_seq.shape
        
        # Embed codebook indices: [batch_size, seq_len] -> [batch_size, seq_len, code_dim]
        embedded = self.codebook_embedding(codebook_seq)
        
        if self.architecture == "mlp":
            # Average pool over sequence: [batch_size, seq_len, code_dim] -> [batch_size, code_dim]
            pooled = embedded.mean(dim=1)
            # Predict: [batch_size, code_dim] -> [batch_size, frame_size * pred_frames]
            pred_motion_flat = self.predictor(pooled)
            
        elif self.architecture == "lstm":
            # LSTM: Sequential processing
            # [batch_size, seq_len, code_dim] -> [batch_size, seq_len, hidden_dim]
            lstm_out, (h_n, c_n) = self.lstm(embedded)
            # Use last hidden state: [batch_size, hidden_dim]
            pooled = lstm_out[:, -1, :]
            # Predict: [batch_size, hidden_dim] -> [batch_size, frame_size * pred_frames]
            pred_motion_flat = self.predictor(pooled)
            
        elif self.architecture == "transformer":
            # Transformer: Self-attention
            # Add positional encoding
            embedded = self.pos_encoding(embedded)
            # [batch_size, seq_len, code_dim] -> [batch_size, seq_len, code_dim]
            transformer_out = self.transformer(embedded)
            # Use last token (most recent context): [batch_size, code_dim]
            pooled = transformer_out[:, -1, :]
            # Predict: [batch_size, code_dim] -> [batch_size, frame_size * pred_frames]
            pred_motion_flat = self.predictor(pooled)
            
        elif self.architecture == "cnn1d":
            # 1D CNN: Convolutional layers
            # Permute for Conv1d: [batch_size, seq_len, code_dim] -> [batch_size, code_dim, seq_len]
            embedded_permuted = embedded.permute(0, 2, 1)
            # Apply CNN: [batch_size, code_dim, seq_len] -> [batch_size, hidden_dim, 1]
            cnn_out = self.cnn1d(embedded_permuted)
            # Squeeze: [batch_size, hidden_dim, 1] -> [batch_size, hidden_dim]
            pooled = cnn_out.squeeze(-1)
            # Predict: [batch_size, hidden_dim] -> [batch_size, frame_size * pred_frames]
            pred_motion_flat = self.predictor(pooled)
        
        # Reshape to [batch_size, pred_frames, frame_size]
        pred_motion = pred_motion_flat.view(batch_size, self.pred_frames, self.frame_size)
        
        return pred_motion


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

