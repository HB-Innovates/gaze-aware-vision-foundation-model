"""Gaze encoder for processing gaze direction vectors."""

import torch
import torch.nn as nn
from typing import List


class GazeEncoder(nn.Module):
    """MLP-based encoder for gaze direction vectors.
    
    Encodes 3D gaze vectors (yaw, pitch, roll) into learned embeddings.
    
    Args:
        input_dim: Dimension of input gaze vector (default: 3)
        embed_dim: Output embedding dimension (default: 512)
        hidden_dims: List of hidden layer dimensions (default: [64, 128, 256])
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        embed_dim: int = 512,
        hidden_dims: List[int] = [64, 128, 256],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final projection to embedding dimension
        layers.extend([
            nn.Linear(prev_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        ])
        
        self.encoder = nn.Sequential(*layers)
        
        # Learnable positional encoding for gaze components
        self.position_embedding = nn.Parameter(torch.randn(1, input_dim))
        
    def forward(self, gaze_vectors: torch.Tensor) -> torch.Tensor:
        """
        Encode gaze direction vectors.
        
        Args:
            gaze_vectors: Gaze vectors [batch_size, 3] (yaw, pitch, roll in radians)
            
        Returns:
            Gaze embeddings [batch_size, embed_dim]
        """
        # Add positional embedding
        gaze_input = gaze_vectors + self.position_embedding
        
        # Encode through MLP
        embeddings = self.encoder(gaze_input)
        
        return embeddings
