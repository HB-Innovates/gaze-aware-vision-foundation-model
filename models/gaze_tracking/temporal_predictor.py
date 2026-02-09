"""Temporal gaze prediction using sequential models."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class TemporalGazePredictor(nn.Module):
    """LSTM-based temporal gaze predictor.
    
    Predicts future gaze directions (1-5 frames ahead) based on
    historical gaze sequence.
    
    Args:
        input_dim: Dimension of gaze vector (default: 3)
        hidden_dim: LSTM hidden dimension (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        prediction_horizon: Number of future frames to predict (default: 5)
        dropout: Dropout rate (default: 0.2)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        prediction_horizon: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Prediction heads for each future frame
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, input_dim),
            )
            for _ in range(prediction_horizon)
        ])
    
    def forward(
        self,
        gaze_sequence: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict future gaze directions.
        
        Args:
            gaze_sequence: Historical gaze vectors [batch_size, seq_len, 3]
            hidden_state: Optional LSTM hidden state (h, c)
            
        Returns:
            predictions: Future gaze predictions [batch_size, prediction_horizon, 3]
            hidden_state: Updated LSTM hidden state (h, c)
        """
        batch_size = gaze_sequence.size(0)
        
        # Embed input sequence
        embedded = self.input_embedding(gaze_sequence)
        
        # Process through LSTM
        if hidden_state is not None:
            lstm_out, hidden_state = self.lstm(embedded, hidden_state)
        else:
            lstm_out, hidden_state = self.lstm(embedded)
        
        # Use last timestep encoding for predictions
        last_encoding = lstm_out[:, -1, :]  # [B, hidden_dim]
        
        # Predict multiple future frames
        predictions = []
        for head in self.prediction_heads:
            pred = head(last_encoding)  # [B, 3]
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)  # [B, prediction_horizon, 3]
        
        return predictions, hidden_state
    
    def predict_next_frame(self, gaze_sequence: torch.Tensor) -> torch.Tensor:
        """
        Predict only the next frame (1 frame ahead).
        
        Args:
            gaze_sequence: Historical gaze vectors [batch_size, seq_len, 3]
            
        Returns:
            Next frame prediction [batch_size, 3]
        """
        predictions, _ = self.forward(gaze_sequence)
        return predictions[:, 0, :]  # Return first prediction
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded temporal predictor from {checkpoint_path}")


class TransformerGazePredictor(nn.Module):
    """Transformer-based temporal gaze predictor.
    
    Alternative to LSTM using transformer architecture for
    better long-range temporal dependencies.
    
    Args:
        input_dim: Dimension of gaze vector (default: 3)
        d_model: Transformer model dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 3)
        prediction_horizon: Number of future frames (default: 5)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        prediction_horizon: int = 5,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Prediction head
        self.predictor = nn.Linear(d_model, input_dim * prediction_horizon)
    
    def forward(self, gaze_sequence: torch.Tensor) -> torch.Tensor:
        """
        Predict future gaze directions using transformer.
        
        Args:
            gaze_sequence: Historical gaze vectors [batch_size, seq_len, 3]
            
        Returns:
            Future predictions [batch_size, prediction_horizon, 3]
        """
        # Project and add positional encoding
        x = self.input_proj(gaze_sequence)  # [B, L, d_model]
        x = self.positional_encoding(x)
        
        # Transform
        encoded = self.transformer(x)  # [B, L, d_model]
        
        # Use last timestep for prediction
        last_state = encoded[:, -1, :]  # [B, d_model]
        
        # Predict all future frames
        predictions = self.predictor(last_state)  # [B, input_dim * horizon]
        predictions = predictions.view(
            -1, self.prediction_horizon, self.input_dim
        )  # [B, horizon, 3]
        
        return predictions


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return x
