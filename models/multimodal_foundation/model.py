"""Main multi-modal gaze-aware vision foundation model."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .vision_encoder import VisionEncoder
from .gaze_encoder import GazeEncoder
from .projection import CrossModalProjection
from .attention import GazeGuidedAttention


class GazeAwareVisionModel(nn.Module):
    """Multi-modal foundation model integrating vision and gaze modalities.
    
    This model combines a vision encoder (CLIP-based) with a gaze encoder,
    using gaze-guided attention mechanisms to enhance visual understanding.
    
    Args:
        vision_embed_dim: Dimension of vision embeddings (default: 512)
        gaze_embed_dim: Dimension of gaze embeddings (default: 512)
        fusion_dim: Dimension of fused embeddings (default: 1024)
        num_attention_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        vision_embed_dim: int = 512,
        gaze_embed_dim: int = 512,
        fusion_dim: int = 1024,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vision_embed_dim = vision_embed_dim
        self.gaze_embed_dim = gaze_embed_dim
        self.fusion_dim = fusion_dim
        
        # Encoders
        self.vision_encoder = VisionEncoder(embed_dim=vision_embed_dim)
        self.gaze_encoder = GazeEncoder(
            input_dim=3,  # (yaw, pitch, roll)
            embed_dim=gaze_embed_dim,
            hidden_dims=[64, 128, 256],
        )
        
        # Cross-modal projection
        self.projection = CrossModalProjection(
            vision_dim=vision_embed_dim,
            gaze_dim=gaze_embed_dim,
            output_dim=fusion_dim,
        )
        
        # Gaze-guided attention
        self.gaze_attention = GazeGuidedAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        # Output projection
        self.output_proj = nn.Linear(fusion_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        images: torch.Tensor,
        gaze_vectors: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the gaze-aware vision model.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            gaze_vectors: Gaze direction vectors [batch_size, 3] (yaw, pitch, roll)
            return_attention: Whether to return attention weights
            
        Returns:
            Fused embeddings [batch_size, fusion_dim]
            Attention weights (optional) [batch_size, num_heads, seq_len, seq_len]
        """
        # Encode vision
        vision_embeds = self.vision_encoder(images)  # [B, vision_embed_dim]
        
        # Encode gaze if provided
        if gaze_vectors is not None:
            gaze_embeds = self.gaze_encoder(gaze_vectors)  # [B, gaze_embed_dim]
        else:
            # Use zero gaze embeddings if not provided
            batch_size = images.shape[0]
            gaze_embeds = torch.zeros(
                batch_size, self.gaze_embed_dim,
                device=images.device, dtype=images.dtype
            )
        
        # Project to fusion space
        fused_embeds = self.projection(vision_embeds, gaze_embeds)  # [B, fusion_dim]
        
        # Apply gaze-guided attention
        attended_embeds, attention_weights = self.gaze_attention(
            fused_embeds,
            gaze_context=gaze_embeds,
            return_attention=return_attention,
        )
        
        # Output projection
        output = self.output_proj(attended_embeds)
        output = self.dropout(output)
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from {checkpoint_path}")
    
    def save_checkpoint(self, checkpoint_path: str, **kwargs):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            **kwargs,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
